import networkx as nx
from collections import defaultdict, Counter
from typing import *
from functools import cmp_to_key
import pickle
import hashlib
import numpy as np
from scipy.sparse import csr_matrix

# 定义优先级 节点和边
NODE_NUMER = {
    'MuSC': 1,
    'Myoblast': 2,
    'Myofiber': 3,
}

# invariants get
def get_cell_invariant(g, node_id: int):
    """
        获取节点不变量cell_type degree
        return:tuple[int, tuple[int]]
    """
    node = g.nodes[node_id]
    cell_type = node['cell_type']
    degree = g.degree(node_id)
    # invariants可扩展....

    type_val = NODE_NUMER.get(cell_type, 9999)
    return (type_val, tuple([degree]))  # 元组确保可哈希和比较
