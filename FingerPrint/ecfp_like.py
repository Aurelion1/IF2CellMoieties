import pickle
import networkx as nx
import zlib
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import *
from dataclasses import dataclass, field

from invariants import CellInvariantExtractor, CELL_TYPE_ENCODING, EDGE_TYPE_ENCODING
import mmh3
# 节点优先级，用于绘制有向边
# 当前一共四种细胞，肌肉干细胞，成肌细胞，肌肉纤维
# 节点序号
# 固定种子
SEED = 42

# mmh3哈希函数
def hash_iteration_input(binary_data: bytes, seed: int = SEED) -> int:
    """
    对二进制数据进行 MurmurHash3 哈希
    
    Args:
        binary_data: 二进制编码的迭代输入
        seed: 哈希种子
        
    Returns:
        int: 32 位无符号整数标识符
    """
    hash_bytes = mmh3.hash_bytes(binary_data, seed=seed)
    identifier = int.from_bytes(hash_bytes, byteorder='little') & 0xffffffff
    return identifier

    
def encode_iteration_input_binary(
    iteration: int,
    core_id: int,
    neighbor_info: List[Tuple[int, int]]
) -> bytes:
    """
    将迭代输入编码为二进制字节串
    
        
    Returns:
        bytes: [iteration, identifier, (bond1, identifier1), ...]
    """
    binary_parts = []
    
    # 1. 迭代层级 (4 字节 uint32)
    binary_parts.append(iteration.to_bytes(4, byteorder='little', signed=False))
    
    # 2. 中心原子标识符 (4 字节，处理负数)
    # 将 32 位有符号整数转换为无符号表示
    core_id_uint = core_id & 0xffffffff
    binary_parts.append(core_id_uint.to_bytes(4, byteorder='little', signed=False))
    
    # 3. 邻居信息 (每个邻居 8 字节：4 字节键类型 + 4 字节邻居 ID)
    for bond_order, neighbor_id in neighbor_info:
        # 键类型 (4 字节 uint32)
        binary_parts.append(bond_order.to_bytes(4, byteorder='little', signed=False))
        # 邻居 ID (4 字节，处理负数)
        neighbor_id_uint = neighbor_id & 0xffffffff
        binary_parts.append(neighbor_id_uint.to_bytes(4, byteorder='little', signed=False))
    
    return b''.join(binary_parts)

def fold_identifier(identifier: int, vector_length: int, seed: int = SEED) -> int:
        """
        二次哈希折叠，将 32 位 ID 映射到指纹位空间
        """
        # MurmurHash3 二次哈希
        hash_value = mmh3.hash(str(identifier).encode('utf-8'), seed=seed)
        # 处理有符号整数，确保非负
        hash_value = hash_value & 0xffffffff
        return hash_value % vector_length
# ============================================================================
# Iterative Updating
# ============================================================================

class IdentifierUpdater:
    """
    Iterative Updating of Identifiers
    
    1. 哈希输入必须包含迭代层级（iteration number）
    2. 邻居必须按 (键级，邻居标识符) 排序
    3. 保留所有中间迭代结果
    """
    
    def __init__(self, edge_type_encoding: Dict[str, int] = None):
        self.edge_type_encoding = edge_type_encoding or EDGE_TYPE_ENCODING
        
    
    def get_edge_code(self, G: nx.Graph, u: Any, v: Any) -> int:
        """获取边的类型编码（对应文献 bond order: 1,2,3,4）"""
        edge_attr = G.edges.get((u, v), {})
        edge_type = edge_attr.get('edge_type', 'default')
        return self.edge_type_encoding.get(edge_type)
    
    def collect_neighbor_info(self, G: nx.Graph, node: Any, 
                              node_identifiers: Dict[Any, int]) -> List[Tuple]:
        """
        收集节点的邻居信息，邻居节点按照键级-标识符大小确定性排序   
        Returns:
            List[Tuple]: 排序后的 [(edge_code, neighbor_id), ...]
        """
        neighbor_info = []
        
        for neighbor in G.neighbors(node):
            edge_code = self.get_edge_code(G, node, neighbor)
            neighbor_id = node_identifiers[neighbor]
            # 文献要求：先边类型，后邻居 ID
            neighbor_info.append((edge_code, neighbor_id))
        
        # 确定性排序，避免顺序依赖（解决不同系统遍历顺序问题）
        neighbor_info.sort()
        
        return neighbor_info
    
    def update_identifier(self, node_id: int, iteration: int, 
                          neighbor_info: List[Tuple]) -> int:
        """
        更新单个节点的标识符
        [n,identifier,bond1,identifier1,bond2,identifier2,...]
        identifier：[1, -1100000244, 1, 1559650422, 1, 1572579716, 2, -1074141656]
        """
        # 哈希输入：(迭代层级，中心 ID，邻居信息元组)
        hash_input = encode_iteration_input_binary(iteration, node_id, neighbor_info)
        new_id = hash_iteration_input(hash_input)
        
        return new_id
    
    def run_iteration(self, G: nx.Graph, node_identifiers: Dict[Any, int], 
                      iteration: int) -> Dict[Any, int]:
        new_identifiers = {}
        # update node identifier
        for node in G.nodes():
            neighbor_info = self.collect_neighbor_info(G, node, node_identifiers)
            new_identifiers[node] = self.update_identifier(
                node_identifiers[node], 
                iteration, 
                neighbor_info
            )
        
        return new_identifiers

# Bond Set
class BondSetTracker:
    def __init__(self, G: nx.Graph, edge_type_encoding: Dict[str, int] = None):
        """
        Args:
            G: 细胞网络图
            edge_type_encoding
        """
        self.G = G
        self.edge_type_encoding = edge_type_encoding or EDGE_TYPE_ENCODING
    
    def normalize_bond(self, u: Any, v: Any, edge_type_code: int) -> Tuple:
        """     
        Returns:
            Tuple: (source_type, target_type, edge_code)
        """
        edge_attr = self.G.edges.get((u, v), {})
        
        tu = self.G.nodes[u].get('cell_type', 'default')
        tv = self.G.nodes[v].get('cell_type', 'default')
        directed = edge_attr['directed']
        if directed:
            # 根据direction映射bond set
            direction = edge_attr['direction']
            if tu == direction[0]:
                return (u, v, edge_type_code)
            else:
                return (v, u, edge_type_code)
        else:
            # 无向边：按节点编号从小到大排序，保证顺序唯一不重复
            if u < v:
                return (u, v, edge_type_code)
            else:
                return (v, u, edge_type_code)  

    
    def get_initial_bond_sets(self) -> Dict[Any, FrozenSet]:
        """
        初始化键集合
        """
        return {node: frozenset() for node in self.G.nodes()}
    
    def update_bond_set(self, node: Any, prev_bond_sets: Dict[Any, FrozenSet],
                        iteration: int) -> FrozenSet:
        """
        更新键集合
        """
        current_bond_set = set(prev_bond_sets[node])
        
        for neighbor in self.G.neighbors(node):
            # 合并邻居的键集合
            current_bond_set.update(prev_bond_sets[neighbor])
            
            # 添加连接边
            edge_code = self.edge_type_encoding.get(
                self.G.edges[(node, neighbor)].get('edge_type', 'default'), 
                0
            )
            bond_tuple = self.normalize_bond(node, neighbor, edge_code)
            current_bond_set.add(bond_tuple)
        
        return frozenset(current_bond_set)
    
    def run_iteration(self, prev_bond_sets: Dict[Any, FrozenSet], 
                      iteration: int) -> Dict[Any, FrozenSet]:
        """执行一轮完整的键集合更新"""
        return {
            node: self.update_bond_set(node, prev_bond_sets, iteration)
            for node in self.G.nodes()
        }


# ============================================================================
# Duplicate Removal
# ============================================================================
@dataclass
class Feature:
    """特征数据结构"""
    center_node: Any
    radius: int
    identifier: int
    bond_set: frozenset = field(default_factory=frozenset)


class StructureDeduplicator:
    def remove_structural_duplicates(self, features: List[Feature]) -> List[Feature]:
        """
        基于键集移除结构重复特征
        """
        unique_features = {}  # Key: bond_set_key, Value: Feature
        
        for feat in features:
            # 将键集合转换为可哈希的元组
            bond_set_key = tuple(sorted(feat.bond_set))
            
            if bond_set_key not in unique_features:
                unique_features[bond_set_key] = feat
            else:
                # 应用文献去重规则
                existing = unique_features[bond_set_key]
                
                # 规则 1: 保留半径小的
                if feat.radius < existing.radius:
                    unique_features[bond_set_key] = feat
                # 规则 2: 半径相同，保留 ID 小的
                elif feat.radius == existing.radius and feat.identifier < existing.identifier:
                    unique_features[bond_set_key] = feat
        
        return list(unique_features.values())
    
    def remove_identifier_duplicates(self, features: List[Feature], 
                                     use_counts: bool = False) -> Dict:
        """
        标识符去重
        Args:
            use_counts: False=二进制指纹，True=计数指纹
        """
        if use_counts:
            identifier_counts = {}
            for feat in features:
                identifier_counts[feat.identifier] = identifier_counts.get(feat.identifier, 0) + 1
            return identifier_counts
        else:
            unique_identifiers = set(feat.identifier for feat in features)
            return list(unique_identifiers)


# ============================================================================
# Fingerprint Generation
# ============================================================================

class CellNetworkECFP:
    """
    细胞网络 ECFP 指纹生成器    
    - radius: 迭代半径
    - useCounts: 是否保留计数信息
    - includeRedundantEnvironments: 是否包含冗余环境
    - nBits: 指纹向量长度
    """
    
    def __init__(self, cell_type_encoding: Dict[str, int] = None,
                 edge_type_encoding: Dict[str, int] = None):
        self.invariant_extractor = CellInvariantExtractor(cell_type_encoding)
        self.identifier_updater = IdentifierUpdater(edge_type_encoding)
        self.bond_set_tracker = None
        self.deduplicator = StructureDeduplicator()
    
    def generate_features(self, G: nx.Graph, radius: int = 2,
                         include_redundant_environments: bool = False) -> List[Feature]:
        """
        生成所有特征（包含去重）
        
        Args:
            G: 细胞网络图
            radius: 迭代半径
            include_redundant_environments: 是否包含冗余环境
            
        Returns:
            List[Feature]: 去重后的特征列表
        """
        self.bond_set_tracker = BondSetTracker(G, self.identifier_updater.edge_type_encoding)
        
        # === 阶段 1: 初始分配 (Initial Assignment) ===
        node_identifiers = self.invariant_extractor.get_initial_identifiers(G)
        bond_sets = self.bond_set_tracker.get_initial_bond_sets()
        
        # 收集初始特征（Radius 0）
        all_features = []
        for node in G.nodes():
            all_features.append(Feature(
                center_node=node,
                radius=0,
                identifier=node_identifiers[node],
                bond_set=bond_sets[node]
            ))
        
        # === 阶段 2: 迭代更新 (Iterative Updating) ===
        for r in range(1, radius + 1):
            # 更新标识符
            node_identifiers = self.identifier_updater.run_iteration(
                G, node_identifiers, r
            )
            # 更新键集合
            bond_sets = self.bond_set_tracker.run_iteration(bond_sets, r)
            
            # 收集当前轮次特征
            for node in G.nodes():
                all_features.append(Feature(
                    center_node=node,
                    radius=r,
                    identifier=node_identifiers[node],
                    bond_set=bond_sets[node]
                ))
        
        # === 阶段 3: 去重处理 (Duplicate Removal) ===
        if not include_redundant_environments:
            # 结构去重
            unique_features = self.deduplicator.remove_structural_duplicates(all_features)
        else:
            # 保留冗余环境
            unique_features = all_features
        
        return unique_features
    
    def generate_fingerprint(self, G: nx.Graph, radius: int = 2,
                            vector_length: int = 1024,
                            use_counts: bool = False,
                            include_redundant_environments: bool = False) -> Tuple:
        """
        生成最终指纹向量
        
        Args:
            G: 细胞网络图
            radius: 迭代半径
            vector_length: 指纹向量长度
            use_counts: 是否使用计数指纹
            include_redundant_environments: 是否包含冗余环境
            
        Returns:
            Tuple: (fingerprint_vector, index_to_feature, feature_info, collision_stats)
        """
        # 生成去重后的特征
        features = self.generate_features(G, radius, include_redundant_environments)
        
        # 标识符去重
        identifiers = self.deduplicator.remove_identifier_duplicates(
            features, use_counts=use_counts
        )
        
        # 生成指纹向量
        fingerprint_vector = [0] * vector_length
        index_to_feature = {}
        feature_info = {}
        collision_stats = defaultdict(int)
        
        if use_counts:
            # 计数指纹
            for ident, count in identifiers.items():
                index = fold_identifier(ident, vector_length)
                if fingerprint_vector[index] > 0:
                    collision_stats[index] += 1
                fingerprint_vector[index] += count
                index_to_feature[index] = ident
        else:
            # 二进制指纹
            for ident in identifiers:
                index = fold_identifier(ident, vector_length)
                if fingerprint_vector[index] == 1:
                    collision_stats[index] += 1
                fingerprint_vector[index] = 1
                index_to_feature[index] = ident
        
        # 建立特征信息映射（对应文献 Interpretation of Identifiers, p.746-747）
        for feat in features:
            feature_info[feat.identifier] = {
                'center_node': feat.center_node,
                'radius': feat.radius,
                'bond_set': feat.bond_set
            }
        
        return fingerprint_vector, index_to_feature, feature_info, dict(collision_stats)
    

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """使用示例"""
    
    # 1. 构建细胞网络图
    G = nx.Graph()
    G.add_node('cell_1', cell_type='MuSC')
    G.add_node('cell_2', cell_type='Myoblast')
    G.add_node('cell_3', cell_type='Myofiber')
    G.add_edge('cell_1', 'cell_2', edge_type='->',directed=False)
    G.add_edge('cell_2', 'cell_3', edge_type='--',directed=False)
    G.add_edge('cell_1', 'cell_3', edge_type='->',directed=False)
    
    # 2. 初始化 ECFP 生成器
    ecfp = CellNetworkECFP(
        cell_type_encoding=CELL_TYPE_ENCODING,
        edge_type_encoding=EDGE_TYPE_ENCODING
    )
    
    # 3. 生成指纹
    fingerprint, index_map, feature_info, collisions = ecfp.generate_fingerprint(
        G, 
        radius=2,
        vector_length=1024,
        use_counts=False
    )
    
    # 4. 查看结果
    print(f"指纹长度：{len(fingerprint)}")
    print(f"非零位数：{sum(fingerprint)}")
    print(f"特征数量：{len(feature_info)}")
    print(f"哈希碰撞数：{sum(collisions.values())}")
    
    # 5. 可解释性：查看某一位对应的子结构
    print("\n指纹位详情 (前 5 位):")
    for bit_idx, ident in list(index_map.items())[:5]:
        info = feature_info.get(ident, {})
        print(f"Bit {bit_idx}: 中心节点={info.get('center_node')}, "
              f"半径={info.get('radius')}, 键集大小={len(info.get('bond_set', []))}")