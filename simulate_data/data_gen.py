'''生成simulate_data'''
import networkx as nx
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import random
import itertools
import os
import pickle
import networkx as nx
from datetime import datetime
# 细胞类型，模拟的身体
CELL_TYPES = ['MuSC', 'Myoblast', 'Myofiber']
color_map = {
        'MuSC': 'blue',
        'Myoblast': 'orange',
        'Myofiber': 'green'
}
# 类型编码
NODE_NUMER = {
    'MuSC': 1,
    'Myoblast': 2,
    'Myofiber': 3,
}
# 键的方向指向：优先级小的指向优先级大的，然后最后端指向初始的细胞。


def generate_random_planar_cell_graph(
    num_nodes: int,
    max_extra_edges: int = 15,
    seed: int = None,
    type_weights: list = [0.5, 0.3, 0.2]
):
    """
    生成带细胞类型的随机平面图（连通、有环、平面）
    :param num_nodes: 节点数量
    :param max_extra_edges: 在树基础上最多加多少条边
    :param seed: 随机种子
    :return: networkx.Graph，每个节点有 'cell_type' 属性
    """
    if seed is not None:
        random.seed(seed)
        # 注意：nx.random_tree 也受全局 random 影响

    # Step 1: 生成随机树（天然平面、连通）
    G = nx.random_tree(num_nodes, seed=seed)

    # Step 2: 为每个节点分配细胞类型
    type_weights = type_weights  # 可调整
    cell_assignments = random.choices(CELL_TYPES, weights=type_weights, k=num_nodes)
    for i, cell_type in enumerate(cell_assignments):
        G.nodes[i]['cell_type'] = cell_type
        G.nodes[i]['cell_weight'] = NODE_NUMER[cell_type]

    # Step 3: 预计算所有非边（避免重复尝试）
    all_non_edges = list(itertools.combinations(G.nodes(), 2))
    non_edges = [e for e in all_non_edges if not G.has_edge(*e)]
    random.shuffle(non_edges)

    # Step 4: 尝试添加边，保持平面性
    added_edges = 0
    for u, v in non_edges:
        if added_edges >= max_extra_edges:
            break
        # 临时加边测试
        G_test = G.copy()
        G_test.add_edge(u, v)
        if nx.check_planarity(G_test)[0]:
            G.add_edge(u, v)
            added_edges += 1

    print(f"Generated graph with {num_nodes} nodes, {G.number_of_edges()} edges, "
          f"{added_edges} extra edges added.")

    return G

def sample_num_nodes():
    """随机选择小图或中图"""
    if random.random() < 0.5:
        return random.randint(20, 49)   # 小图
    else:
        return random.randint(50, 70)   # 中图

# 使用环结构替换图中原有的节点
def replace_node_with_myo_cycle(G, node):
    """
    用 MuSC -> Myoblast -> Myofiber -> MuSC 的三角环替换指定节点。
    所有原邻居连接到新的 MuSC 节点（作为接口），保证平面性。
    """
    if node not in G:
        raise ValueError(f"Node {node} not in graph")
    
    # 1. 保存原邻居
    neighbors = list(G.neighbors(node))
    
    # 2. 删除原节点
    G.remove_node(node)
    
    # 3. 生成新节点 ID（确保唯一）
    existing_nodes = set(G.nodes())
    base_id = max(existing_nodes) + 1 if existing_nodes else 0
    msc = base_id
    myoblast = base_id + 1
    myofiber = base_id + 2

    # 4. 添加新节点（带新细胞类型）
    G.add_node(msc, cell_type='MuSC', cell_weight=NODE_NUMER['MuSC'])
    G.add_node(myoblast, cell_type='Myoblast', cell_weight=NODE_NUMER['Myoblast'])
    G.add_node(myofiber, cell_type='Myofiber', cell_weight=NODE_NUMER['Myofiber'])

    # 5. 构建内部环（无向图，所以边是双向的）
    G.add_edges_from([
        (msc, myoblast),
        (myoblast, myofiber),
        (myofiber, msc)
    ])

    # 6. 将所有原邻居连接到 MuSC（单一接口，保平面）
    for nb in neighbors:
        G.add_edge(nb, msc)

    return G

def has_cycle(G):
    """
    检查图中是否有环
    """
    return nx.cycle_basis(G)

def no_Mycycle(G, u, v, cell_types):
    """
    检查添加边 (u, v) 是否会形成 {MuSC, Myoblast, Myofiber} 三角形。
    """
    if not G.has_node(u) or not G.has_node(v):
        return False

    # 获取 u 和 v 的共同邻居
    common_neighbors = set(G.neighbors(u)) & set(G.neighbors(v))
    target_set = {'MuSC', 'Myb', 'Myf'}

    for w in common_neighbors:
        types = {cell_types[u], cell_types[v], cell_types[w]}
        if types == target_set:
            return True
    return False


def has_forbidden_myogenic_triangle(G):
    """
    高效检查图 G 中是否存在 {MuSC, Myoblast, Myofiber} 三角形。
    使用邻居对遍历法，适用于稀疏图（如平面图）。
    """
    target_set = {'MuSC', 'Myb', 'Myf'}
    try:
        cell_types = nx.get_node_attributes(G, 'cell_type')
    except KeyError:
        raise ValueError("All nodes must have 'cell_type' attribute.")

    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        n = len(neighbors)
        for i in range(n):
            for j in range(i + 1, n):
                v, w = neighbors[i], neighbors[j]
                if G.has_edge(v, w):  # (u, v, w) 构成三角形
                    types = {cell_types[u], cell_types[v], cell_types[w]}
                    if types == target_set:
                        return True, [u, v, w]
    return False, None
# annotate edge directions
# 赋予边方向,按照节点的优先级,优先级小的指向优先级大的,然后最后端指向初始的细胞。1->2->3->1
def annotate_edge_directions(G, seed=None):
    """
    同类节点之间的边：标记为无向（directed=False）
    异类节点之间的边：标记方向（directed=True, direction=(from_type, to_type)）
        
    Return:G
    """
    G_annotated = G.copy()
    weights = {G.nodes[n]["cell_weight"] for n in G.nodes}
    min_w, max_w = min(weights), max(weights)

    for u, v in G_annotated.edges():
        tu = G_annotated.nodes[u]["cell_type"]
        tv = G_annotated.nodes[v]["cell_type"]
        
        if tu is None or tv is None:
            raise ValueError(f"Node {u} or {v} missing 'cell_type' attribute.")
        
        if tu == tv:
            # 同类边：无向
            G_annotated[u][v]['directed'] = False
            G_annotated[u][v]['direction'] = None
        else:
            # 异类边：根据cell_weight确定方向
            wu = G_annotated.nodes[u]["cell_weight"]
            wv = G_annotated.nodes[v]["cell_weight"]
            
            # 优先级小的指向优先级大的，然后最后端指向初始的细胞
            # MuSC(1) -> Myoblast(2) -> Myofiber(3) -> MuSC(1)
            if wu == max_w and wv == min_w:
                src, dst = u, v
            elif wu == min_w and wv == max_w:
                src, dst = v, u

            # ===== 规则 A：线性优先级 =====
            elif wu < wv:
                src, dst = u, v
            else:
                src, dst = v, u
            G_annotated[u][v]["directed"] = True
            G_annotated[u][v]["direction"] = (
                G_annotated.nodes[src]["cell_type"],
                G_annotated.nodes[dst]["cell_type"],
            )
            
    
    return G_annotated

# visual
def visual_graph(DG):
    # 传入的是无向图
    pos = nx.spring_layout(DG)
    
    # 颜色映射
    plt.figure(figsize=(8, 6), dpi=400)
    colors = [color_map.get(DG.nodes[n].get('cell_type', 'unknown'), 'lightgray') for n in DG.nodes()]
    # 绘制节点
    nx.draw_networkx_nodes(DG, pos, node_color=colors, node_size=40)
    # 绘制节点标签
    nx.draw_networkx_labels(DG, pos, font_size=5, font_color='black')
    # 绘制无向边
    undirected_edges = [(u, v) for u, v in DG.edges() if not DG[u][v].get('directed', False)]
    nx.draw_networkx_edges(DG, pos, edgelist=undirected_edges, edge_color='gray', width=1.5, alpha=0.5)
    
    # 可视化带有方向的边 [(1,3),(1,4)....]
    directed_edges = [(u, v) for u, v in DG.edges() if DG[u][v].get('directed', False)]
    
    # 获取边的方向  start->end
    dirs = [DG[u][v].get('direction', False) for u, v in directed_edges]
    
    # 调整边的方向
    for i, (u, v) in enumerate(directed_edges):
        # 获取边两端节点的类型
        start = DG.nodes[u].get('cell_type', 'unknown')
        end = DG.nodes[v].get('cell_type', 'unknown')
        # 是否和dirs[i]中记录的方向一致
        if (start, end) != dirs[i]:
            directed_edges[i] = (v, u)
    
    # 在图上画出有向边，箭头表示(利用临时有向图表示)
    temp_dg = nx.DiGraph()
    temp_dg.add_edges_from(directed_edges)
    
    # 在临时有向图上绘制有向边
    nx.draw_networkx_edges(temp_dg, pos, edgelist=directed_edges,
                           edge_color='black', arrows=True, arrowsize=4,
                           )  
    # 获取图中的所有 myogenic 环
    myo_cycles = get_myo_cycles(DG)
    
    # 特殊标识符合的环结构的边
    for cycle in myo_cycles:
        cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)] + [(cycle[-1], cycle[0])]
        nx.draw_networkx_edges(DG, pos, edgelist=cycle_edges, edge_color='red', width=2.5, alpha=0.7)
    # 添加节点图例
    legend_elements = [Patch(facecolor=color, label=label) for label, color in color_map.items()]
    plt.legend(handles=legend_elements, title="Node Types")
    plt.show()


# 环结构获取以及环结构判断和可视化
def get_myo_cycles(DG):
    """
    获取图中所有人工嵌入的 myogenic 环
    返回：列表，每个元素是一个环（节点 ID 列表）
    """
    cycles = []
    # 判断规则，由于节点之间的边类型固定，
    # 所以环中只能包含 MuSC, Myoblast, Myofiber 三种类型的节点，由于有边的指向规则，所以只要是这样的环结构就行
    for cycle in nx.cycle_basis(DG):
        cycle_types = [DG.nodes[node]['cell_type'] for node in cycle]
        if sorted(cycle_types) == sorted(['MuSC', 'Myoblast', 'Myofiber']):
            # 检查环的方向是否正确
                cycles.append(cycle)
    return cycles
