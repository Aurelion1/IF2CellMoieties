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
