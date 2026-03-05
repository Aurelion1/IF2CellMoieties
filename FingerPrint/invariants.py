"""
细胞节点不变量提取模块
对应文献：Initial Assignment of Atom Identifiers (p.743)
"""
import zlib
from typing import Any, Dict, Tuple, Union
import networkx as nx
import mmh3
# 细胞类型编码（可配置）
CELL_TYPE_ENCODING = {
    'Musc': 1,      # 肌肉干细胞
    'Myb': 2,       # 成肌细胞
    'Myf': 3,       # 肌肉纤维
}
# 边类型
EDGE_TYPE_ENCODING = {
    '->': 1,
    '--': 2,
    '-|': 3,
    '|-': 4,
    '<->': 5,
    'default': 0
}

class CellInvariantExtractor:
    """细胞节点不变量提取器"""
    
    def __init__(self, cell_type_encoding: Dict[str, int] = None):
        """
        初始化不变量提取器
        
        Args:
            cell_type_encoding: 细胞类型编码字典，若为 None 则使用默认编码
        """
        self.cell_type_encoding = cell_type_encoding or CELL_TYPE_ENCODING
    
    def get_node_invariants(self, G: nx.Graph, node: Any) -> Tuple:
        """
        获取单个节点的不变量元组
        1. 细胞类型编码
        2. 总连接度（degree）
        3. 邻居数
        4. 是否在环中（可选）
        
        Returns:
            Tuple: 不变量元组，用于后续哈希
        """
        node_attr = G.nodes[node]
        
        invariants = (
            self.cell_type_encoding.get(
                node_attr.get('cell_type', 'default'), 
            ),
            G.degree(node),                              # 总连接度
            len(list(G.neighbors(node))),                # 邻居数量
            nx.cycle_basis(G.subgraph([node] + list(G.neighbors(node)))) 
                and 1 or 0                               # 是否在环中（简化判断）
        )
        
        return invariants
    
    def invariants_to_fixed_binary(self, invariants: Tuple) -> bytes:
        """
        convert invariants to fixed length binary
        Args:
            invariants: 4元素不变量元组 (细胞类型编码, 总连接度, 邻居数, 是否在环中)
        Returns:
            bytes: 固定6字节的二进制串（little端序，无符号）
        """
        cell_type_code, degree, neighbor_count, in_cycle = invariants
        # 1. 细胞类型编码：1字节 uint8（little端序）
        cell_type_bin = cell_type_code.to_bytes(
            length=1, 
            byteorder='little', 
            signed=False
        )
        # 2. 总连接度：2字节 uint16（little端序）
        degree_bin = degree.to_bytes(
            length=2, 
            byteorder='little', 
            signed=False
        )
        # 3. 邻居数：2字节 uint16（little端序）
        neighbor_bin = neighbor_count.to_bytes(
            length=2, 
            byteorder='little', 
            signed=False
        )
        # 4. 是否在环中：1字节 uint8（0/1）
        in_cycle_bin = in_cycle.to_bytes(
            length=1, 
            byteorder='little', 
            signed=False
        )
        # 拼接所有二进制字段（固定顺序，保证唯一性）
        fixed_binary = cell_type_bin + degree_bin + neighbor_bin + in_cycle_bin
        return fixed_binary
    def hash_invariants(self, G:nx.Graph,invariants: Tuple) -> int:
        """
        将不变量元组哈希为 32 位整数标识符
        
        Returns:
            int: 32 位无符号整数标识符
        """
        return zlib.crc32(str(invariants).encode('utf-8')) & 0xffffffff
    
    def get_initial_identifiers(self, G: nx.Graph,seed:int = 42) -> Dict[Any, int]:
        """
        get initial identifier for node
        Returns:
            Dict: {node: identifier}
        """
        identifiers = {}
        for node in G.nodes():
            invariants = self.get_node_invariants(G, node)
            binary_invariants = self.invariants_to_fixed_binary(invariants)
            hash_bytes = mmh3.hash_bytes(binary_invariants, seed=seed)
            identifier = int.from_bytes(hash_bytes, byteorder='little') & 0xffffffff
            identifiers[node] = identifier
        return identifiers

