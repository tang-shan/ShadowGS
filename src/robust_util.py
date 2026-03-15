import numpy as np
from pycocotools import mask as mask_util
import torch
from PIL import Image
import cv2


def segment_overlap_separate(mask, segments, min_area=0, overlap_threshold=0.5,binary_threshold=0.5):
    segment_rles = [s['segmentation'] for s in segments]
    segment_areas = [s['area'] for s in segments]

    mask_np = mask.detach().cpu().numpy()
    mask_np = (mask_np > binary_threshold).astype(np.uint8)

    mask_rle = mask_util.encode(np.asfortranarray(mask_np))
    intersections = [mask_util.merge([mask_rle, segment_rle], intersect=1) for segment_rle in segment_rles]
    areas_overlaps = [mask_util.area(rle) / seg_area for (rle, seg_area) in zip(intersections, segment_areas)]

    # 分别保存每个选中的segment
    selected_masks = []
    for i, (seg, seg_area, overlap) in enumerate(zip(segment_rles, segment_areas, areas_overlaps)):
        if seg_area < min_area:
            continue
        if overlap >= overlap_threshold:
            decoded_mask = mask_util.decode(seg)
            selected_masks.append(decoded_mask)
    
    # 如果你想要所有选中的segments但不合并，可以返回列表
    return selected_masks

def merge_masks(masks, default_shape=(900, 1600)):
    """
    更健壮的合并函数，处理各种边界情况
    输入: masks list, 每个元素是numpy array
    """
    if not masks:
        return np.zeros(default_shape, dtype=np.uint8)
    
    # 检查所有mask的形状是否一致
    first_shape = masks[0].shape
    for i, mask in enumerate(masks):
        if mask.shape != first_shape:
            print(f"警告: mask {i} 的形状 {mask.shape} 与第一个mask {first_shape} 不一致")
    
    # 初始化合并mask (numpy array)
    merged_mask = np.zeros_like(masks[0], dtype=np.bool_)
    valid_count = 0
    
    for mask in masks:
        if mask is not None:
            # 处理数据类型 (numpy版本)
            if mask.dtype != np.bool_:
                mask_bool = (mask > 0)
            else:
                mask_bool = mask
            
            # 合并 (numpy逻辑或)
            merged_mask = np.logical_or(merged_mask, mask_bool)
            valid_count += 1
    
    if valid_count == 0:
        return np.zeros(default_shape, dtype=np.uint8)
    
    return merged_mask.astype(np.uint8)


def save_binary_mask_pil(mask, filename):
    """
    使用 PIL 保存二值掩码（支持 torch tensor）
    """
    # 如果是 torch tensor，转换为 numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # 确保是 2D 数组
    if mask.ndim == 3:
        mask = mask.squeeze()  # 去除通道维度
    
    img = Image.fromarray(mask, mode='L')
    img.save(filename)

import numpy as np
import cv2
from pycocotools import mask as mask_util

def process_binary_mask_to_coco(binary_mask, category_id=1, image_id=1, annotation_id=1, min_area=50):
    """
    安全的二值掩码处理函数 - 使用pycocotools mask格式
    """
    try:
        # 确保输入是numpy数组
        if not isinstance(binary_mask, np.ndarray):
            binary_mask = np.array(binary_mask)
        
        # 检查是否真的全黑
        if np.max(binary_mask) == 0:
            return []
        
        # 转为真正的二值图 (0和255) - pycocotools需要这种格式
        if binary_mask.dtype == np.uint8 and np.max(binary_mask) == 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        elif np.max(binary_mask) > 1 and np.max(binary_mask) < 255:
            # 如果是其他范围的灰度图，转为0-255
            binary_mask = (binary_mask > 0).astype(np.uint8) * 255
        elif np.max(binary_mask) == 255:
            # 已经是0-255格式，确保类型正确
            binary_mask = binary_mask.astype(np.uint8)
        
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        binary_mask_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 使用清理后的掩码进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_clean)
        
        annotations = []
        valid_count = 0
        
        for i in range(1, num_labels):  # 跳过背景(0)
            area = stats[i, cv2.CC_STAT_AREA]
            bbox = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]  # [x, y, w, h]
            
            # 过滤条件
            if area < min_area:
                continue
                
            # 检查宽高比
            width, height = bbox[2], bbox[3]
            if min(width, height) == 0:  # 避免除零
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 50:
                continue
                
            # 创建单个对象的掩码
            obj_mask = (labels == i).astype(np.uint8) * 255
            
            # 使用pycocotools转换为RLE格式
            rle = mask_util.encode(np.asfortranarray(obj_mask))
            
            # 计算面积和边界框
            area = mask_util.area(rle).item()
            bbox = mask_util.toBbox(rle).tolist()
            
            annotation = {
                'id': annotation_id + valid_count,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': rle,  # RLE格式
                'area': area,
                'bbox': bbox,  # [x, y, width, height]
                'iscrowd': 0
            }
            annotations.append(annotation)
            valid_count += 1
        
        return annotations
        
    except Exception as e:
        print(f"处理二值掩码时出错: {e}")
        import traceback
        traceback.print_exc()
        return []  # 确保返回空列表而不是None

def image_to_coco_mask(image, use_otsu=True, threshold=127):
    """
    将图像转为适合COCO处理的二值掩码 (0和255)
    """
    # 转为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 二值化
    if use_otsu:
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_mask.astype(np.uint8)

"""
def shadow_overlap(shadow_mask, shadow_coco):
    masks = segment_overlap_separate(shadow_mask, shadow_coco)
    
    # 检查 masks 是否为空列表
    if len(masks) == 0:
        # 直接在 GPU 上创建全零 tensor，保持相同的设备和数据类型
        combine_mask = torch.zeros_like(shadow_mask, dtype=torch.uint8)
        return combine_mask
    
    combine_mask = np.max(masks, axis=0).astype(np.uint8)
    combine_mask = torch.from_numpy(combine_mask)
    combine_mask = combine_mask.cuda()
    return combine_mask
"""

def shadow_overlap(shadow_mask, shadow_coco,overlap_threshold=0.5):
    """最小修改版本 - 保持原有逻辑但修复内存问题"""
    if shadow_mask.dtype != torch.bool:
        shadow_mask = shadow_mask.bool()
    masks = segment_overlap_separate(shadow_mask, shadow_coco,overlap_threshold)
    
    if len(masks) == 0:
        return torch.zeros_like(shadow_mask, dtype=torch.bool)
    
    # 修复：使用torch操作替代numpy
    mask_tensors = []
    for mask_np in masks:
        mask_tensor = torch.from_numpy(mask_np).to(shadow_mask.device).bool()
        mask_tensors.append(mask_tensor)
    
    # 在GPU上进行合并操作
    if mask_tensors:
        combine_mask = mask_tensors[0]
        for mask in mask_tensors[1:]:
            combine_mask = combine_mask | mask
        return combine_mask
    else:
        return torch.zeros_like(shadow_mask, dtype=torch.bool)


def shadow_overlap_indoor(shadow_mask, shadow_coco,overlap_threshold=0.5):
    """最小修改版本 - 保持原有逻辑但修复内存问题"""
    if shadow_mask.dtype != torch.bool:
        shadow_mask = shadow_mask.bool()
    #shadow_mask = ~shadow_mask
    masks = segment_overlap_separate(shadow_mask, shadow_coco,overlap_threshold)
    
    if len(masks) == 0:
        return torch.zeros_like(shadow_mask, dtype=torch.bool)
    
    # 修复：使用torch操作替代numpy
    mask_tensors = []
    for mask_np in masks:
        mask_tensor = torch.from_numpy(mask_np).to(shadow_mask.device).bool()
        mask_tensors.append(mask_tensor)
    
    # 在GPU上进行合并操作
    if mask_tensors:
        combine_mask = mask_tensors[0]
        for mask in mask_tensors[1:]:
            combine_mask = combine_mask | mask
        return combine_mask
    else:
        return torch.zeros_like(shadow_mask, dtype=torch.bool)

"""
# 修正调用部分
# 读取网络预测的mask，并转换为torch tensor
network_mask = cv2.imread('test_mask.jpg', cv2.IMREAD_GRAYSCALE)
network_mask = torch.tensor(network_mask).float() / 255.0  # 转为0-1范围的torch tensor

# 读取shadow mask，并转换为COCO格式
shadow_img = cv2.imread('./data/dog/all_shadow_masks/0001.png', cv2.IMREAD_GRAYSCALE)
shadow_img = image_to_coco_mask(shadow_img)
shadow_coco = process_binary_mask_to_coco(shadow_img)


mask = shadow_overlap(network_mask,shadow_coco)
"""

