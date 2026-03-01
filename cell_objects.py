import os
import warnings
from functools import partial
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,binary_dilation
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from skimage import io, measure, morphology
from skimage.measure import euler_number
from skimage.morphology import disk, erosion
from tqdm import tqdm
import scipy.ndimage
from typing import *
CELL_TYPES = ['MuSC', 'Myoblast', 'Myofiber']

NODE_NUMER = {
    'MuSC': 1,
    'Myoblast': 2,
    'Myofiber': 3,
}

def shrink_mask_iterative(mask, n_pixels=5):
    """
    Iterative, topology-preserving shrink.
    Similar to CellProfiler 'Shrink objects by a specified number of pixels'.
    """
    current = mask.copy()
    selem = disk(1)

    for _ in range(n_pixels):
        eroded = erosion(current, selem)

        # 防止完全消失
        if eroded.sum() == 0:
            break
        # 拓扑保护（防断裂/打洞）
        if euler_number(eroded) != euler_number(current):
            break
        current = eroded

    return current

class MyofiberObject:
    def __init__(self, region, shrink_pixels=8):
        """
        region: skimage.measure._regionprops.RegionProperties
        """
        # ===== 基本属性 =====
        self.id = region.label
        self.area = region.area
        self.centroid = region.centroid          # (row, col)
        self.bbox = region.bbox                  # (minr, minc, maxr, maxc)
        # 肌肉纤维的区域坐标
        self.minr, self.minc, self.maxr, self.maxc = self.bbox

        # ===== 局部 mask（只覆盖 bbox）=====
        # shape = (maxr-minr, maxc-minc)
        self.local_mask = region.image.astype(bool)
        # ===== 内部区域（CellProfiler 风格 shrink）=====
        self.internal_local_mask = shrink_mask_iterative(
            self.local_mask,
            n_pixels=shrink_pixels
        )

        # ===== 成肌细胞相关属性 =====
        self.is_myoblast = False
        self.contained_nuclei_ids = []
        self.contained_nuclei_count = 0

        # ===== 空间索引用（可选）=====
        self.center_point = (self.centroid[1], self.centroid[0])  # (x, y)
    # --------------------------------------------------
    # 传入的是细胞核经过扩展后的区域
    def nucleus_overlap_ratio(self, nucleus_object):
        """
        返回 nucleus 在 internal 区域内的比例（0~1）
        """
        nuc_minr, nuc_minc, nuc_maxr, nuc_maxc = nucleus_object.expanded_bbox
        # 计算全图坐标系下的bbox交集
        inter_minr = max(self.minr, nuc_minr)
        inter_minc = max(self.minc, nuc_minc)
        inter_maxr = min(self.maxr, nuc_maxr)
        inter_maxc = min(self.maxc, nuc_maxc)
        # 检查bbox是否有交集
        if inter_minr >= inter_maxr or inter_minc >= inter_maxc:
            return 0.0
        # 3. 从肌纤维 internal mask 中提取交集区域
        fiber_in_inter = self.internal_local_mask[
            inter_minr - self.minr : inter_maxr - self.minr,
            inter_minc - self.minc : inter_maxc - self.minc
        ]

        # 4. 从细胞核 expanded mask 中提取交集区域
        nucleus_in_inter = nucleus_object.expanded_local_mask[
            inter_minr - nuc_minr : inter_maxr - nuc_minr,
            inter_minc - nuc_minc : inter_maxc - nuc_minc
        ]
        # 5. 计算重叠
        overlap = np.sum(fiber_in_inter & nucleus_in_inter)
        nucleus_total_area = np.sum(nucleus_object.expanded_local_mask)

        if nucleus_total_area == 0:
            return 0.0

        return overlap / nucleus_total_area


    # --------------------------------------------------
    def __repr__(self):
        return (
            f"Myofiber {self.id}: "
            f"Area={self.area}, "
            f"IsMyoblast={self.is_myoblast}, "
            f"Nuclei={self.contained_nuclei_count}"
        )

def expand_single_object(mask, expand_px):
    if expand_px <= 0:
        return mask

    dist = distance_transform_edt(~mask)
    return dist <= expand_px
# 扩展函数存在一个问题，扩展后，细胞核区域应该大于当前的边界，但是目前并没有
class NucleusObject:
    def __init__(self, original_region, expanded_region):
        """
        region    : skimage.measure.RegionProperties
        expand_px : 扩展像素数（CellProfiler ExpandObjects 语义）
        """

        # ===== 基本属性 =====
        self.id = original_region.label
        self.area = original_region.area
        self.centroid = (original_region.centroid[1], original_region.centroid[0])  # (x, y)
        self.bbox = original_region.bbox  # (minr, minc, maxr, maxc)

        self.expanded_region = expanded_region
        self.expanded_bbox = expanded_region.bbox  # (minr, minc, maxr, maxc)
        
        self.minr, self.minc, self.maxr, self.maxc = self.bbox
        self.expanded_minr, self.expanded_minc, self.expanded_maxr, self.expanded_maxc = self.expanded_bbox
        
        # ===== 局部 mask（bbox 内，已经是精确连通域）=====
        self.local_mask = original_region.image.astype(bool)
        self.expanded_local_mask = self.expanded_region.image.astype(bool)


        # ===== 关系属性（后续填充）=====
        self.is_boundary = False
        self.belong_myofiber_id = None

        # ===== KDTree / 空间索引用 =====
        self.center_point = self.centroid


    # --------------------------------------------------
    def __repr__(self):
        return (
            f"Nucleus {self.id}: "
            f"Area={self.area}, "
            f"Centroid={self.centroid}, "
            f"IsBoundary={self.is_boundary}"
        )
# 提取MuSC的mask
class MuSCObject:
    def __init__(self, original_region, expanded_region):
        """
        MuSC（肌卫星细胞）对象类，与 NucleusObject 结构保持一致
        :param original_region: skimage.measure.RegionProperties 原始区域
        :param expanded_region: skimage.measure.RegionProperties 膨胀/扩张后区域
        """
        # ===== 基础属性（与细胞核类保持统一格式）=====
        self.id = original_region.label  # 唯一ID
        self.area = original_region.area  # 原始面积
        self.centroid = (original_region.centroid[1], original_region.centroid[0])  # (x, y) 坐标
        self.bbox = original_region.bbox  # 边界框 (minr, minc, maxr, maxc)

        self.expanded_region = expanded_region
        self.expanded_bbox = expanded_region.bbox

        self.minr, self.minc, self.maxr, self.maxc = self.bbox
        self.expanded_minr, self.expanded_minc, self.expanded_maxr, self.expanded_maxc = self.expanded_bbox

        # ===== 掩码区域（原始 + 膨胀后）=====
        self.local_mask = original_region.image.astype(bool)  # 局部小掩码
        self.expanded_local_mask = expanded_region.image.astype(bool)  # 膨胀后局部掩码

    def __repr__(self):
        return (
            f"MuSC {self.id}: "
            f"Area={self.area}, "
            f"Centroid={self.centroid}, "
        )


def _dilate_mask_and_bbox(mask: np.ndarray, bbox: Tuple[int, int, int, int], distance: int = 8) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    对局部掩膜进行膨胀，并更新对应的边界框坐标。
    修正点：对掩膜进行填充，确保 mask 形状与 bbox 范围一致。
    """
    if not np.any(mask):
        h, w = mask.shape
        minr, minc, maxr, maxc = bbox
        # 如果 bbox 已更新过，这里需要重新计算预期形状，但通常空掩膜不需要膨胀
        return mask, bbox
    
    # 1. 创建填充后的画布 (Pad the canvas)
    # 原始形状
    h, w = mask.shape
    # 新形状：上下左右各增加 distance
    new_h, new_w = h + 2 * distance, w + 2 * distance
    padded_mask = np.zeros((new_h, new_w), dtype=bool)
    
    # 将原始掩膜放置在中心
    padded_mask[distance:h+distance, distance:w+distance] = mask
    
    # 2. 形态学膨胀
    structure = disk(distance)
    dilated_mask = binary_dilation(padded_mask, structure=structure)
    
    # 3. 更新边界框坐标 (向外扩展 distance)
    minr, minc, maxr, maxc = bbox
    new_bbox = (
        minr - distance,
        minc - distance,
        maxr + distance,
        maxc + distance
    )
    
    
    return dilated_mask, new_bbox

def _check_mask_intersection(mask1: np.ndarray, bbox1: Tuple[int, int, int, int], 
                             mask2: np.ndarray, bbox2: Tuple[int, int, int, int]) -> bool:
    """
    检查两个带有坐标信息的局部掩膜是否在全局空间中存在像素交集。
    
    :param mask1: 对象 1 的膨胀掩膜
    :param bbox1: 对象 1 的全局边界框 (minr, minc, maxr, maxc)
    :param mask2: 对象 2 的膨胀掩膜
    :param bbox2: 对象 2 的全局边界框
    :return: 是否存在交集 (bool)
    """
    # 1. 计算边界框的全局交集
    inter_minr = max(bbox1[0], bbox2[0])
    inter_minc = max(bbox1[1], bbox2[1])
    inter_maxr = min(bbox1[2], bbox2[2])
    inter_maxc = min(bbox1[3], bbox2[3])
    
    # 2. 若边界框无交集，则掩膜必然无交集
    if inter_minr >= inter_maxr or inter_minc >= inter_maxc:
        return False
    
    # 3. 计算交集区域在各自局部掩膜中的切片索引
    # 对象 1 切片
    s1_r_start = inter_minr - bbox1[0]
    s1_r_end = inter_maxr - bbox1[0]
    s1_c_start = inter_minc - bbox1[1]
    s1_c_end = inter_maxc - bbox1[1]
    
    # 对象 2 切片
    s2_r_start = inter_minr - bbox2[0]
    s2_r_end = inter_maxr - bbox2[0]
    s2_c_start = inter_minc - bbox2[1]
    s2_c_end = inter_maxc - bbox2[1]
    
    # 4. 提取交集区域并判断像素重叠
    region1 = mask1[s1_r_start:s1_r_end, s1_c_start:s1_c_end]
    region2 = mask2[s2_r_start:s2_r_end, s2_c_start:s2_c_end]
    
    return np.any(region1 & region2)

def create_combined_graph(myofiber_objects: List[Any], MuSC_objects: List[Any]) -> nx.Graph:
    """
    构建包含肌纤维、成肌细胞及肌卫星细胞的组合空间关系图。
    
    建模规则：
    1. 每个对象为一个节点。
    2. 若两个对象的掩膜区域各自扩张 8 像素后存在非空交集，则建立边。
    
    :param myofiber_objects: List[MyofiberObject]
    :param MuSC_objects: List[MuSCObject]
    :return: networkx.Graph
    """
    G = nx.Graph()
    
    # ================= 1. 构建节点 =================
    objects = myofiber_objects + MuSC_objects
    id = 1
    for obj in objects:
        if isinstance(obj, MyofiberObject):
            cell_type = "Myoblast" if obj.is_myoblast else "Myofiber"
        elif isinstance(obj, MuSCObject):
            cell_type = "MuSC"

        G.add_node(
            id, 
            cell_type=cell_type, 
            area=obj.area, 
            centroid=obj.centroid, 
            ref=obj # 保留对象引用以便后续访问详细属性
        )
        id += 1
        
    # 扩展掩码构建边关系
    all_nodes = list(G.nodes(data=True))
    n_nodes = len(all_nodes)
    
    spatial_cache = {}
    
    for node_id, data in all_nodes:
        obj = data['ref']
        obj_type = data['cell_type']
        
        # 确定基础掩膜 (使用最具空间代表性的掩膜
        base_mask = obj.local_mask
        bbox = obj.bbox
            
        # 执行扩张操作 (规则：扩张 8 像素)
        dil_mask, dil_bbox = _dilate_mask_and_bbox(base_mask, bbox, distance=8)
        spatial_cache[node_id] = (dil_mask, dil_bbox)
    
    # 两两比对建立边
    for i in range(n_nodes):
        id_i, data_i = all_nodes[i]
        mask_i, bbox_i = spatial_cache[id_i]
        
        for j in range(i + 1, n_nodes):
            id_j, data_j = all_nodes[j]
            mask_j, bbox_j = spatial_cache[id_j]
            
            # 检查空间交集
            if _check_mask_intersection(mask_i, bbox_i, mask_j, bbox_j):
                G.add_edge(id_i, id_j)
                
    return G


def expand_labels_globally(label_image, distance):
    """
    cellprofiler-like：全局扩展，处理冲突
    """
    if distance <= 0:
        return label_image
    
    background = (label_image == 0)
    # 计算距离和最近邻索引
    distances, (i, j) = scipy.ndimage.distance_transform_edt(
        background, return_indices=True
    )
    
    out_labels = label_image.copy()
    
    # 只有在距离范围内且是背景的地方才进行赋值
    mask = background & (distances <= distance)
    
    # 将最近邻的 label 赋给背景像素
    out_labels[mask] = label_image[i[mask], j[mask]]
    
    return out_labels


def identify_myoblasts(
    myofiber_objects,
    nucleus_objects,
    ratio_threshold=0.7
):
    """
    使用 internal mask + nucleus overlap 判定成肌细胞
    """

    print("开始识别成肌细胞（central nuclei）")

    for fiber in myofiber_objects:
        fiber.contained_nuclei_ids = []
        fiber.contained_nuclei_count = 0
        fiber.is_myoblast = False
        # expanded_local_mask只是
        for nuc_object in nucleus_objects:
            # 计算 nucleus 在 fiber internal 区域的重叠比例
            ratio = fiber.nucleus_overlap_ratio(nuc_object)

            if ratio >= ratio_threshold:
                fiber.contained_nuclei_ids.append(nuc_object.id)
                fiber.contained_nuclei_count += 1
                nuc_object.belong_myofiber_id = fiber.id

        # 只要有 ≥1 个中央核，即为成肌细胞
        if fiber.contained_nuclei_count > 0:
            fiber.is_myoblast = True

    myoblast_count = sum(f.is_myoblast for f in myofiber_objects)

    print(f"成肌细胞（myoblast fibers）: {myoblast_count} / {len(myofiber_objects)}")

    return myoblast_count



def _dilate_mask_and_bbox(mask: np.ndarray, bbox: Tuple[int, int, int, int], distance: int = 8) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    对局部掩膜进行膨胀，并更新对应的边界框坐标。
    修正点：对掩膜进行填充，确保 mask 形状与 bbox 范围一致。
    """
    if not np.any(mask):
        h, w = mask.shape
        minr, minc, maxr, maxc = bbox
        # 如果 bbox 已更新过，这里需要重新计算预期形状，但通常空掩膜不需要膨胀
        return mask, bbox
    
    # 1. 创建填充后的画布 (Pad the canvas)
    # 原始形状
    h, w = mask.shape
    # 新形状：上下左右各增加 distance
    new_h, new_w = h + 2 * distance, w + 2 * distance
    padded_mask = np.zeros((new_h, new_w), dtype=bool)
    
    # 将原始掩膜放置在中心
    padded_mask[distance:h+distance, distance:w+distance] = mask
    
    # 2. 形态学膨胀
    structure = disk(distance)
    dilated_mask = binary_dilation(padded_mask, structure=structure)
    
    # 3. 更新边界框坐标 (向外扩展 distance)
    minr, minc, maxr, maxc = bbox
    new_bbox = (
        minr - distance,
        minc - distance,
        maxr + distance,
        maxc + distance
    )
    
    
    return dilated_mask, new_bbox

def _check_mask_intersection(mask1: np.ndarray, bbox1: Tuple[int, int, int, int], 
                             mask2: np.ndarray, bbox2: Tuple[int, int, int, int]) -> bool:
    """
    检查两个带有坐标信息的局部掩膜是否在全局空间中存在像素交集。
    
    :param mask1: 对象 1 的膨胀掩膜
    :param bbox1: 对象 1 的全局边界框 (minr, minc, maxr, maxc)
    :param mask2: 对象 2 的膨胀掩膜
    :param bbox2: 对象 2 的全局边界框
    :return: 是否存在交集 (bool)
    """
    # 1. 计算边界框的全局交集
    inter_minr = max(bbox1[0], bbox2[0])
    inter_minc = max(bbox1[1], bbox2[1])
    inter_maxr = min(bbox1[2], bbox2[2])
    inter_maxc = min(bbox1[3], bbox2[3])
    
    # 2. 若边界框无交集，则掩膜必然无交集
    if inter_minr >= inter_maxr or inter_minc >= inter_maxc:
        return False
    
    # 3. 计算交集区域在各自局部掩膜中的切片索引
    # 对象 1 切片
    s1_r_start = inter_minr - bbox1[0]
    s1_r_end = inter_maxr - bbox1[0]
    s1_c_start = inter_minc - bbox1[1]
    s1_c_end = inter_maxc - bbox1[1]
    
    # 对象 2 切片
    s2_r_start = inter_minr - bbox2[0]
    s2_r_end = inter_maxr - bbox2[0]
    s2_c_start = inter_minc - bbox2[1]
    s2_c_end = inter_maxc - bbox2[1]
    
    # 4. 提取交集区域并判断像素重叠
    region1 = mask1[s1_r_start:s1_r_end, s1_c_start:s1_c_end]
    region2 = mask2[s2_r_start:s2_r_end, s2_c_start:s2_c_end]
    
    return np.any(region1 & region2)

def create_combined_graph(myofiber_objects: List[Any], MuSC_objects: List[Any]) -> nx.Graph:
    """
    构建包含肌纤维、成肌细胞及肌卫星细胞的组合空间关系图。
    
    建模规则：
    1. 每个对象为一个节点。
    2. 若两个对象的掩膜区域各自扩张 8 像素后存在非空交集，则建立边。
    
    :param myofiber_objects: List[MyofiberObject]
    :param MuSC_objects: List[MuSCObject]
    :return: networkx.Graph
    """
    G = nx.Graph()
    
    # ================= 1. 构建节点 =================
    objects = myofiber_objects + MuSC_objects
    id = 1
    for obj in objects:
        if isinstance(obj, MyofiberObject):
            cell_type = "Myoblast" if obj.is_myoblast else "Myofiber"
        elif isinstance(obj, MuSCObject):
            cell_type = "MuSC"

        G.add_node(
            id, 
            cell_type=cell_type, 
            area=obj.area, 
            centroid=obj.centroid, 
            ref=obj # 保留对象引用以便后续访问详细属性
        )
        id += 1
        
    # 扩展掩码构建边关系
    all_nodes = list(G.nodes(data=True))
    n_nodes = len(all_nodes)
    
    spatial_cache = {}
    
    for node_id, data in all_nodes:
        obj = data['ref']
        obj_type = data['cell_type']
        
        # 确定基础掩膜 (使用最具空间代表性的掩膜
        base_mask = obj.local_mask
        bbox = obj.bbox
            
        # 执行扩张操作 (规则：扩张 8 像素)
        dil_mask, dil_bbox = _dilate_mask_and_bbox(base_mask, bbox, distance=8)
        spatial_cache[node_id] = (dil_mask, dil_bbox)
    
    # 两两比对建立边
    for i in range(n_nodes):
        id_i, data_i = all_nodes[i]
        mask_i, bbox_i = spatial_cache[id_i]
        
        for j in range(i + 1, n_nodes):
            id_j, data_j = all_nodes[j]
            mask_j, bbox_j = spatial_cache[id_j]
            
            # 检查空间交集
            if _check_mask_intersection(mask_i, bbox_i, mask_j, bbox_j):
                G.add_edge(id_i, id_j)
                
    return G

def annotate_edge_directions(G, seed=None):
    """
    同类节点之间的边：标记为无向（directed=False）
    异类节点之间的边：标记方向（directed=True, direction=(from_type, to_type)）
        
    Return:G
    """
    G_annotated = G.copy()
    weights = {NODE_NUMER[G.nodes[n]["cell_type"]] for n in G.nodes}
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
            # 异类边：根据cell_type确定方向
            wu = NODE_NUMER[tu]
            wv = NODE_NUMER[tv]
            
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