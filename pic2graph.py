import os
from collections import defaultdict
from skimage import io
import numpy as np
from skimage import measure, io, morphology
import networkx as nx
import pickle
from typing import *
from cell_objects import *

imgs_pth = r"E:/1/2026.1.1-2026.1.5/if_image/results"
output_pth = r"/result"
os.makedirs(output_pth, exist_ok=True)

# 获取所有文件名
imgs_names = os.listdir(imgs_pth)
print(f"共找到 {len(imgs_names)} 个文件")

# 按 组织_块号 分组，每组内再按通道分类
groups = defaultdict(lambda: {'DAPI': None, 'laminin': None, 'MuSC': None, 'others': []})

for img_name in imgs_names:
    # 解析文件名: {组织}_{块号}_{通道}.tif
    parts = img_name.replace('.tiff', '').replace('.tif', '').split('_')
    print(parts)
    if len(parts) >= 3:
        tissue = parts[0]
        block = parts[1]
        channel = parts[2]  # DAPI / laminin / MuSC
        
        group_key = f"{tissue}_{block}"
        print(channel)
        # 分类到对应通道
        if channel in ['DAPI', 'laminin', 'MuSC']:
            groups[group_key][channel] = img_name
        else:
            groups[group_key]['others'].append(img_name)
    else:
        print(f"跳过不符合格式的文件: {img_name}")

print(f"\n共分成 {len(groups)} 组")

# 检查每组完整性
complete_groups = []
for key, channels in sorted(groups.items()):
    missing = []
    for ch in ['DAPI', 'laminin', 'MuSC']:
        if channels[ch] is None:
            missing.append(ch)
    
    if not missing:
        complete_groups.append(key)

print(f"\n完整组数: {len(complete_groups)}/{len(groups)}")

# ================= 按组进行分析 =================

for group_key in complete_groups:
    print(f"\n{'='*50}")
    print(f"分析组: {group_key}")
    print(f"{'='*50}")
    
    channels = groups[group_key]
    
    # 加载三个通道
    dapi_path = os.path.join(imgs_pth, channels['DAPI'])
    laminin_path = os.path.join(imgs_pth, channels['laminin'])
    musc_path = os.path.join(imgs_pth, channels['MuSC'])
    
    print(f"加载 DAPI: {channels['DAPI']}")
    DNA = io.imread(dapi_path)  # 细胞核
    
    print(f"加载 laminin: {channels['laminin']}")
    Myf = io.imread(laminin_path)  # 成肌纤维（基底膜）
    
    print(f"加载 MuSC: {channels['MuSC']}")
    MuSC = io.imread(musc_path)  # 肌肉干细胞
    
    # 识别MuSC，Myofiber，Myoblast
    MuSC = np.array(MuSC)
    myofibers = np.array(Myf)
    DNA = np.array(DNA)
    # 标记连通域
    # myofiber
    myofibers_label = measure.label(myofibers,connectivity=2)
    myofiber_regions = measure.regionprops(myofibers_label)
    myofiber_objects = [MyofiberObject(region) for region in myofiber_regions]
    # nucleus
    nuclei_label = measure.label(DNA,connectivity=2)
    nuclei_regions = measure.regionprops(nuclei_label)
    expanded_nuclei_label = expand_labels_globally(nuclei_label, distance=1)
    # 创建Nucleusobjects
    ori_regions = measure.regionprops(nuclei_label)
    expanded_regions = measure.regionprops(expanded_nuclei_label)
    expanded_region_map = {reg.label: reg for reg in expanded_regions}
    nuclei_objects = []
    for original_region in ori_regions:
        label_id = original_region.label
        
        if label_id not in expanded_region_map:
            continue  # 防止丢失
        expanded_region = expanded_region_map[label_id]
        
        # 创建对象（原始 + 对应ID扩张）
        obj = NucleusObject(original_region, expanded_region)
        nuclei_objects.append(obj)
    # 区分myofiber和myoblast
    result = identify_myoblasts(
        myofiber_objects,
        nuclei_objects,
        ratio_threshold=0.85
    )
    # MuSC
    muscs_label = measure.label(MuSC,connectivity=2)
    musc_regions = measure.regionprops(muscs_label)
    # expand
    expanded_muscs_label = expand_labels_globally(muscs_label, distance=1)
    expanded_muscs_regions = measure.regionprops(expanded_muscs_label)

    # 创建MuSCobjects
    expanded_muscs_region_map = {reg.label: reg for reg in expanded_muscs_regions}
    MuSC_objects = []
    for original_region in musc_regions:
        label_id = original_region.label
        if label_id not in expanded_muscs_region_map:
            continue  
        expanded_region = expanded_muscs_region_map[label_id]
        
        # 创建对象（原始 + 对应ID扩张）
        obj = MuSCObject(original_region, expanded_region)
        MuSC_objects.append(obj)
    # 构建合并图
    graph = create_combined_graph(myofiber_objects, MuSC_objects)

    # 查看图基本信息
    print(f"节点总数：{graph.number_of_nodes()}")
    print(f"边总数：{graph.number_of_edges()}")
    # 保存为pkl
    annotate_edge_directions(graph)
    with open(os.path.join(output_pth, f"{group_key}_graph.pkl"), "wb") as f:
        pickle.dump(graph, f)


