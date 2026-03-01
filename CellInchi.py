import networkx as nx
from collections import defaultdict, Counter
from typing import *
from functools import cmp_to_key
import pickle
import hashlib
import numpy as np
from scipy.sparse import csr_matrix

####
from utils import (get_cell_invariant, 
                   find_all_cycles,
                   extract_invariant_cycles,
                   cycle_identifier,
                   reduce_graph_by_cycles,
)

# validate_graph实现需要检查图是否包含必要属性，这个今晚补充
### Cell inchi类

class InChI_new:
    def __init__(self, graph: nx.Graph or nx.DiGraph):
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise TypeError("Graph must be a NetworkX Graph/DiGraph")
        self.graph = graph.copy()
        self._validate_graph()
        self._is_directed = graph.is_directed()
        # tmp
        self.count = 0
        # canonicalization/intermediate state
        self.initial_invariants: Dict[int, Tuple] = {}
        self.sorted_nodes: List[int] = []
        self.initial_ranks: Dict[int, int] = {}
        self.num_diff_initial_ranks: int = 0
        self.last_refined_colors: Dict[int, int] = {}

        # 保存 UpdateFullLinearCT 的结果：每个连通分量 -> {'start':node, 'lct': tuple((u,v),...)}
        # lct 使用 canonical ids (1-based)
        self.full_linear_ct: List[Dict] = []

    def _validate_graph(self) -> None:
        for node in self.graph.nodes():
            if 'cell_type' not in self.graph.nodes[node]:
                raise ValueError(f"Node {node} missing 'cell_type' attribute")
        for u, v in self.graph.edges():
            if 'edge_type' not in self.graph[u][v]:
                self.graph[u][v]['edge_type'] = '-'

    def _comp_invariants_only(self, node1: int, node2: int) -> int:
        inv1 = self.initial_invariants[node1]
        inv2 = self.initial_invariants[node2]
        if inv1 < inv2:
            return -1
        elif inv1 > inv2:
            return 1
        return 0

    def _comp_invariants(self, node1: int, node2: int) -> int:
        r = self._comp_invariants_only(node1, node2)
        if r != 0:
            return r
        return -1 if node1 < node2 else 1

    def _sort_nodes_by_invariants(self, nodes: List[int]) -> List[int]:
        return sorted(nodes, key=cmp_to_key(self._comp_invariants))

    def _compute_initial_ranks(self,component: nx.Graph) -> None:
        nodes = list(component.nodes())
        if not nodes:
            self.sorted_nodes = []
            self.initial_ranks = {}
            self.num_diff_initial_ranks = 0
            return
        self.initial_invariants = {node: get_cell_invariant(component, node) for node in nodes}
        self.sorted_nodes = self._sort_nodes_by_invariants(list(nodes))
        # print(self.initial_invariants)
        self.initial_ranks = {}
        self.num_diff_initial_ranks = 1
        num_nodes = len(self.sorted_nodes)
        current_rank = num_nodes
        last_node = self.sorted_nodes[-1]
        self.initial_ranks[last_node] = current_rank
        # self.graph.nodes[last_node]['initial_rank'] = current_rank
        for i in range(num_nodes - 2, -1, -1):
            current_node = self.sorted_nodes[i]
            next_node = self.sorted_nodes[i + 1]
            if self._comp_invariants_only(current_node, next_node) != 0:
                self.num_diff_initial_ranks += 1
                current_rank = i + 1
            self.initial_ranks[current_node] = current_rank
            # self.graph.nodes[current_node]['initial_rank'] = current_rank
    
    def build_node_to_id_from_order(self, order: List[int]) -> Dict[int, int]:
            return {node: i + 1 for i, node in enumerate(order)}

    def _get_initial_colors(self) -> Dict[int, int]:
        return self.initial_ranks.copy()
    # 颜色初始化 重写为兼容连通分量
    def _refine_partition(self, graph: nx.Graph, current_colors: Dict[int, int], max_iter: int = 20) -> Dict[int, int]:
        nodes = list(graph.nodes())
        # print("nodes:", nodes)
        for _ in range(max_iter):
            signature_groups: DefaultDict[Tuple, List[int]] = defaultdict(list)
            for node in nodes:
                neighbor_colors = sorted(current_colors[nbr] for nbr in graph.neighbors(node))
                signature = (current_colors[node], tuple(neighbor_colors))
                signature_groups[signature].append(node)
            sorted_signatures = sorted(signature_groups.keys())
            new_colors: Dict[int, int] = {}
            cumulative_count = 0
            for sig in sorted_signatures:
                group_nodes = signature_groups[sig]
                group_size = len(group_nodes)
                group_color = cumulative_count + group_size
                for node in group_nodes:
                    new_colors[node] = group_color
                cumulative_count += group_size
            if new_colors == current_colors:
                break
            current_colors = new_colors
        return current_colors


    def _sort_nodes_by_type_blocks(self, graph: nx.Graph, colors: Dict[str, int]) -> List[int]:
        nodes = list(graph.nodes())
        type_groups: DefaultDict[str, List[int]] = defaultdict(list)
        for node in nodes:
            cell_type = graph.nodes[node]['cell_type']
            type_groups[cell_type].append(node)
        sorted_types = sorted(type_groups.keys())
        result: List[int] = []
        for cell_type in sorted_types:
            nodes = sorted(type_groups[cell_type], key=lambda n: (colors[n], n))
            result.extend(nodes)
        # print("sorted_types",sorted_types)
        # print("result",result)
        return result

    # 连接表构造
    # -------- linear CT 的构造与比较工具 --------
    def _ct_build_partial(self, subgraph: nx.Graph, fixed_nodes: List[int], colors: Dict[int, int]) -> Tuple[int, ...]:
        """
        构建部分连接表（用于剪枝）
        按颜色升序排列已固定的节点
        格式：节点颜色 + 小于该颜色的邻居颜色（已固定节点间的连接）
        """
        # 将固定节点按颜色升序排序
        sorted_fixed = sorted(fixed_nodes, key=lambda n: colors[n])
        
        connection_table = []
        fixed_set = set(fixed_nodes)
        
        for node in sorted_fixed:
            current_color = colors[node]
            
            # 获取与已固定节点中颜色小于当前节点的邻居颜色
            neighbor_colors = []
            for nbr in subgraph.neighbors(node):
                if (nbr in fixed_set and 
                    colors[nbr] < current_color):
                    neighbor_colors.append(colors[nbr])
            # 排序邻居颜色
            neighbor_colors.sort()
            
            # 添加到连接表
            connection_table.append(current_color)
            connection_table.extend(neighbor_colors)
        
        return tuple(connection_table)

    def _ct_build_full(self, subgraph: nx.Graph, colors: Dict[int, int]) -> Tuple[int, ...]:
        """
        构建完整连接表（用于最终比较）
        按颜色升序排列所有节点
        格式：节点颜色 + 小于该颜色的所有邻居颜色
        """
        # 从colors[node,color]中获取节点，按颜色升序排列所有节点
        all_nodes = list(subgraph.nodes())
        # print("all_nodes",all_nodes)
        sorted_nodes = sorted(all_nodes, key=lambda n: colors[n])
        
        connection_table = []
        
        for node in sorted_nodes:
            current_color = colors[node]
            
            # 获取所有颜色小于当前节点的邻居颜色
            neighbor_colors = []
            for nbr in subgraph.neighbors(node):
                if colors[nbr] < current_color:
                    neighbor_colors.append(colors[nbr])
            # 排序邻居颜色
            neighbor_colors.sort()
            
            # 添加到连接表
            connection_table.append(current_color)
            connection_table.extend(neighbor_colors)
        
        return tuple(connection_table)

    def _ct_compare(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        """
        合并的比较函数，处理所有比较场景
        返回：-1(a<b), 0(a==b), 1(a>b)
        """
        la, lb = len(a), len(b)
        lm = min(la, lb)
        
        for i in range(lm):
            if a[i] < b[i]:
                return -1
            if a[i] > b[i]:
                return 1
        
        # 前缀相同，较短者更小
        if la < lb:
            return -1
        if la > lb:
            return 1
        
        return 0
    # 同构群和轨道，暂时没加入代码流程中
    def _init_orbit(self, nodes: List[int]):
        self.orbit = {n: n for n in nodes}
        self.generators = []

    def _update_orbit(self, sigma: Dict[int, int]):
        # sigma: mapping from one ordering to another (node->node)
        changed = True
        while changed:
            changed = False
            for u in list(self.orbit.keys()):
                v = sigma.get(u, u)
                old = self.orbit[u]
                new = min(self.orbit[u], self.orbit.get(v, v))
                if new != old:
                    self.orbit[u] = new
                    changed = True
    # -------- symmetry breaking using linearCT --------
    def _break_symmetry(self,   graph: nx.Graph,
                                current_colors: Dict[str, int],
                                fixed_nodes: List[int] = None,
                                best_full_ct: Tuple[int, ...] = None) -> Tuple[List[int], Tuple[int, ...]]:
        """
        非平凡层的递归搜索
        """
        if fixed_nodes is None:
            fixed_nodes = []
        
        # 检查是否所有颜色唯一
        if len(set(current_colors.values())) == len(current_colors):
            # order = self._sort_nodes_by_type_blocks(graph,current_colors)
            order = current_colors
            # print("current_colors",order)
            full_ct = self._ct_build_full(graph,current_colors)
            return order, full_ct
        
        # 找到最小重复颜色
        color_groups = defaultdict(list)
        for node, color in current_colors.items():
            color_groups[color].append(node)
        
        duplicate_colors = [c for c, nodes in color_groups.items() if len(nodes) > 1]
        
        target_color = min(duplicate_colors)
        all_colors = sorted(color_groups.keys())
        target_idx = all_colors.index(target_color)
        
        if target_idx == 0:
            new_color = 1
        else:
            # 设置为前一颜色+1
            prev_color = all_colors[target_idx - 1]
            new_color = prev_color + 1
        
        candidates = sorted(color_groups[target_color])
        
        best_order = None
        best_ct = best_full_ct
        
        for cand in candidates:
            new_colors = current_colors.copy()
            new_colors[cand] = new_color
            
            refined_colors = self._refine_partition(graph,new_colors)
            new_fixed = fixed_nodes + [cand]
            
            # 剪枝：使用合并的比较函数
            if best_ct is not None:
                partial_ct = self._ct_build_partial(graph, new_fixed, refined_colors)
                
                # 如果部分连接表已经比当前最佳完整连接表大，剪枝
                if self._ct_compare(partial_ct, best_ct) > 0:
                    continue
            
            # 递归
            try_order, try_ct = self._break_symmetry(graph,refined_colors, new_fixed, best_ct)
            
            if try_order is not None:
                if best_ct is None or self._ct_compare(try_ct, best_ct) < 0:
                    best_ct = try_ct
                    best_order = try_order
                # 自同构群后面完善
                # elif self._ct_compare(try_ct, best_ct) == 0:
                    # 如果连接表相同，说明在此分区中是一个自同构群
                    # if best_order is not None and try_order is not None:
                    #     sigma = {best_order[i]: try_order[i] for i in range(len(best_order))}
                    #     self.generators.append(sigma)
                    #     self._update_orbit(sigma)
        
        return best_order, best_ct
    

    # ================= main API =================
    # 按照能够处理连通分量来重写代码逻辑，联通分量排序按照full LCT字典序来进行排序；最后按排序后的顺序依次拼接成全图的 canonical labeling
    # component处理：分别对图的component（联通分量）进行处理，每个component独立计算canonical labeling，最后按component顺序拼接。
    def get_canonical_order(self) -> List[int]:
        if self.graph.number_of_nodes() == 0:
            return []
        # component处理，获取component数
        # result保存每个component的canonical order和full LCT
        results = []
        components = list(nx.connected_components(self.graph))
        for component in components:
            subgraph = self.graph.subgraph(component)
            # print("component:", subgraph)
            self._compute_initial_ranks(subgraph)
            initial_colors = self._get_initial_colors()
            # print("initial_colors:", initial_colors)
            # 初始化颜色
            refined_colors = self._refine_partition(subgraph,initial_colors)
            self.last_refined_colors = refined_colors.copy()
            # 返回每个联通分量component的canonical order和full LCT
            canonical_order,best_ct = self._break_symmetry(subgraph,refined_colors)
            canonical_order = sorted(canonical_order.items(), key=lambda item: item[1])
            # print("canonical_order:", canonical_order)
            results.append((canonical_order,best_ct,component))
        # print("canonical_order:", canonical_order)
        return results
    # 根据component的连接表确定各component的规范化顺序
    def _sort_components_by_ct(self, results_components: List[Tuple[Dict[str,int], Tuple[int, ...], Set[int]]]) -> List[Set[int]]:
        sorted_components = sorted(results_components, key=lambda x: x[1])
        # print("sorted_components:", len(sorted_components), sorted_components)
        return sorted_components
    
    # -------- InChI-like serialization of connectivity (/c) --------
    def _serialize_single_component(self, graph: nx.Graph, component_colors: Dict[str, int], component_nodes_set: Set[str]) -> Tuple[str, str, str]: # -> (comp_c_str, comp_i_str, comp_formula)
        """
        为单个连通分量生成其内部的 c 层、i 层和化学式。
        返回: (connectivity_string, interaction_string, formula_string)
        """
        # print(f"Serializing single component: nodes={component_nodes_set}")
        
        # --- 为每个component生成c层---
        sorted_nodes_in_comp = sorted(component_nodes_set, key=lambda n: dict(component_colors)[n])
        # print(f"  sorted_nodes_in_comp: {sorted_nodes_in_comp}")

        if len(sorted_nodes_in_comp) == 1:
            # Single node component: its canonical ID within *this* component is 1
            local_id = 1
            comp_c_str = str(local_id)
        else:
            # Multi-node component: Need to build adjacency and create a path/string
            # Create a mapping from node to its *local* canonical ID within this component (starting from 1)
            node_to_local_id = {node: i + 1 for i, node in enumerate(sorted_nodes_in_comp)}
            local_id_to_node = {v: k for k, v in node_to_local_id.items()} # Reverse mapping
            # print(f'  local_id_to_node: {local_id_to_node}')
            # Build adjacency list for *this* component using local IDs
            local_adj: Dict[int, List[int]] = defaultdict(list)
            for u, v in graph.edges(): # Use the *component's* subgraph to find connections
                if u in node_to_local_id and v in node_to_local_id:
                    uid = node_to_local_id[u]
                    vid = node_to_local_id[v]
                    # Add both directions for undirected graph
                    local_adj[uid].append(vid)
                    local_adj[vid].append(uid)

            # --- Simplified linear walk for multi-node component connectivity string ---
            # Find the node with the smallest canonical color (local ID 1 after sorting by color)
            start_local_id = 1 # The node with the smallest color in this component gets local ID 1
            visited: Set[int] = set()
            path: List[int] = []
            stack = [start_local_id]

            while stack:
                current_id = stack.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)
                path.append(current_id)

                # Get neighbors and sort them by their canonical order (local ID) to ensure deterministic path
                neighbors = sorted(local_adj[current_id])
                # Add neighbors to the front of the stack (DFS behavior)
                for nb_id in reversed(neighbors):
                    if nb_id not in visited:
                        stack.append(nb_id)

            # Build the connectivity string for this multi-node component
            # This is a simplified version, InChI handles branches/parentheses differently
            comp_c_str_parts = [str(local_id) for local_id in path]
            comp_c_str = "-".join(comp_c_str_parts)

        # --- 2. Generate Interactions (/i) for this component (using local IDs) ---
        comp_interactions: List[Tuple[int, str, int]] = []
        for u, v, data in graph.edges(data=True):
            uid = node_to_local_id.get(u)
            vid = node_to_local_id.get(v)
            if uid is not None and vid is not None: # Ensure local ID was found
                inter_type = data.get('edge_type', '-')
                # Ensure consistent ordering (smaller ID first)
                if uid < vid:
                    comp_interactions.append((uid, inter_type, vid))
                else:
                    comp_interactions.append((vid, inter_type, uid))

        comp_interactions.sort(key=lambda x: (x[0], x[2], x[1]))
        comp_i_str_parts = [f"{u}{t}{v}" for u, t, v in comp_interactions]
        comp_i_str = ",".join(comp_i_str_parts) if comp_i_str_parts else ""

        # --- 3. Generate Formula for this component ---
        comp_cell_types = [graph.nodes[node]['cell_type'] for node in component_nodes_set]
        comp_counter = Counter(comp_cell_types)
        sorted_types = sorted(comp_counter.keys())
        comp_formula_parts = []
        for t in sorted_types:
            count = comp_counter[t]
            if count > 1:
                comp_formula_parts.append(f"{t}{count}")
            else:
                comp_formula_parts.append(t)
        comp_formula = "".join(comp_formula_parts)

        return comp_c_str, comp_i_str, comp_formula

    # -------- Combine results from all components --------
    def _combine_component_results(self, sorted_components_data: List[Tuple[str, str, str]]) -> Tuple[str, str, str]: # -> (overall_formula, overall_connectivity, overall_interaction)
        """
        Combine the results from _serialize_single_component for all sorted components.
        Uses ';' to separate components within a layer, '.' for formula, '/' for overall layers.
        """
        if not sorted_components_data:
            return "", "", ""

        comp_formulas = []
        comp_c_strings = []
        comp_i_strings = []

        for comp_c_str, comp_i_str, comp_formula in sorted_components_data:
            comp_formulas.append(comp_formula)
            comp_c_strings.append(comp_c_str)
            if comp_i_str: # Only add non-empty interaction strings
                comp_i_strings.append(comp_i_str)

        # Join component formulas with '.'
        overall_formula = ".".join(comp_formulas)
        # Join component connectivity strings with ';'
        overall_connectivity = "c" + ";".join(comp_c_strings)
        # Join component interaction strings with ';'
        
        overall_interaction = "i" + ";".join(comp_i_strings) if comp_i_strings else ""

        return overall_formula, overall_connectivity, overall_interaction

    # 生成CellInChI字符串
    def to_cellinchi(self) -> str:
        if self.graph.number_of_nodes() == 0:
            return "CellInChI=1S//"

        print("-----start canonical order--------")
        canonical_order_result = self.get_canonical_order() # This returns List[Tuple[Dict[str, int], Tuple[int, ...], Set[str]]]
        print("-----end canonical order--------")

        if not canonical_order_result: # Check if the result list is empty
            return "CellInChI=1S//"

        # Sort components based on their connection tables
        sorted_components = self._sort_components_by_ct(canonical_order_result)
        # print("sorted_components:", len(sorted_components), sorted_components)

        # --- Serialize each component ---
        serialized_components_data = []
        for component_colors, full_linear_ct, component_nodes_set in sorted_components:
            subgraph = self.graph.subgraph(component_nodes_set)
            comp_c_str, comp_i_str, comp_formula = self._serialize_single_component(subgraph, component_colors, component_nodes_set)
            serialized_components_data.append((comp_c_str, comp_i_str, comp_formula))
            # print(f"  Serialized component: c='{comp_c_str}', i='{comp_i_str}', formula='{comp_formula}'")

        # --- Combine the results from all components ---
        overall_formula, overall_connectivity, overall_interaction = self._combine_component_results(serialized_components_data)

        # --- Collect other layers (e.g., reduction) if needed ---
        # This part is tricky as reduce_graph_by_cycles might need global IDs.
        # You might need to run the global ID assignment first or modify the function.
        # For now, let's assume it's handled separately or is not critical.
        reduction_layer = "" # Placeholder
        try:
            # Example: You might need to assign global IDs here first if reduce_graph_by_cycles requires them
            # global_node_to_id_map = self._assign_global_ids(sorted_components) # Implement if needed
            # reduction_layer = reduce_graph_by_cycles(self.graph, global_node_to_id_map) 
            pass
        except Exception as e:
            print(f"Error in reduce_graph_by_cycles (optional): {e}")

        # --- Build final layers list ---
        main_layers = [layer for layer in [overall_connectivity, overall_interaction] if layer]
        all_layers = main_layers + ([reduction_layer] if reduction_layer and reduction_layer.strip() else [])

        # --- Construct final CellInChI string ---
        # Note: The formula is built from components but represents the whole structure.
        # InChI uses '/' to separate layers.
        return f"CellInChI=1S/{overall_formula}/{'/'.join(all_layers)}"
        



    def get_G(self):
        return self.graph