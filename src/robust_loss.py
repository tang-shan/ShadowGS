import torch
import math
import numpy as np
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torchvision import transforms
from PIL import Image
import os



def read_image_as_tensor(image_path, H=None, W=None, interpolation=transforms.InterpolationMode.BILINEAR):
    """
    将图像读取为PyTorch张量，支持指定尺寸调整
    
    参数:
        image_path (str): 图像文件路径
        H (int, optional): 调整后的图像高度，需为正整数
        W (int, optional): 调整后的图像宽度，需为正整数
        interpolation (InterpolationMode, optional): 插值方法，默认双线性插值
        
    返回:
        torch.Tensor: 形状为(3, H, W)或(3, 原始高度, 原始宽度)、值在0-1之间的张量
        
    异常:
        FileNotFoundError: 图像文件不存在
        ValueError: 输入的H或W不是正整数
        RuntimeError: 图像读取或转换失败
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 检查H和W的有效性
    if (H is not None and (not isinstance(H, int) or H <= 0)) or \
       (W is not None and (not isinstance(W, int) or W <= 0)):
        raise ValueError("H和W必须是正整数")
    
    # 构建转换管道
    transform_steps = []
    
    # 添加 resize 操作（如果指定了H和W）
    if H is not None and W is not None:
        transform_steps.append(transforms.Resize((H, W), interpolation=interpolation))
    
    # 添加转张量和归一化操作
    transform_steps.append(transforms.ToTensor())
    
    # 组合转换操作
    transform = transforms.Compose(transform_steps)
    
    try:
        # 读取并转换图像
        with Image.open(image_path) as img:
            # 确保图像为RGB格式
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            # 处理带alpha通道的图像
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
                
            tensor = transform(img)
            
    except Exception as e:
        raise RuntimeError(f"图像处理失败: {str(e)}") from e
    
    # 验证输出张量
    assert tensor.ndim == 3 and tensor.shape[0] == 3, \
        f"张量形状应为(3, H, W)，实际为{tensor.shape}"
    assert torch.all((tensor >= 0) & (tensor <= 1)), \
        f"张量值应在[0,1]范围内，实际范围: [{tensor.min():.4f}, {tensor.max():.4f}]"
    
    return tensor

def save_tensor_as_image(tensor, output_path):
    """
    将PyTorch张量保存为图像文件，根据张量维度自动判断为灰度图或彩色图
    
    参数:
        tensor (torch.Tensor): 输入张量，支持形状为[1, h, w]、[h, w]或[3, h, w]
        output_path (str): 输出图像文件路径（需包含文件名和扩展名）
        
    异常:
        ValueError: 输入张量维度不支持
        RuntimeError: 图像保存失败
    """
    # 检查输入张量的有效性
    if tensor.dim() not in (2, 3):
        raise ValueError(f"不支持的张量维度: {tensor.dim()}，支持的维度为2或3")
    
    # 处理不同维度的张量
    if tensor.dim() == 3:
        if tensor.shape[0] not in (1, 3):
            raise ValueError(f"3维张量的第一维度必须是1或3，实际为{tensor.shape[0]}")
        
        # 如果是单通道灰度图，转换为2维张量以便保存
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # 移除通道维度，变为[h, w]
    
    # 确保张量值在合理范围内，并转换为0-255的uint8类型
    # 处理可能的归一化情况（如果张量值在0-1之间）
    if tensor.max() <= 1.0 and tensor.min() >= 0.0:
        tensor = tensor * 255.0
    
    # 转换为uint8类型
    tensor = tensor.to(torch.uint8)
    
    # 将张量转换为PIL图像
    try:
        # 对于2维张量（灰度图），使用L模式；3维张量（彩色图）使用RGB模式
        if tensor.dim() == 2:
            img = Image.fromarray(tensor.cpu().numpy(), mode='L')
        else:  # 3维，形状为[3, h, w]
            # 转换为[h, w, 3]并调整通道顺序为RGB
            img = Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), mode='RGB')
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        img.save(output_path)
        
    except Exception as e:
        raise RuntimeError(f"图像保存失败: {str(e)}") from e
    

def rgb_to_brightness(img):
    """
    将RGB图像转换为亮度图
    输入: torch tensor [3, H, W] 或 [H, W, 3]
    """
    # 检查输入形状
    if len(img.shape) == 3 and img.shape[0] == 3:
        # [3, H, W] 格式
        r, g, b = img[0], img[1], img[2]
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # [H, W, 3] 格式  
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        raise ValueError(f"不支持的图像形状: {img.shape}")
    
    # 使用标准的亮度转换公式
    # Y = 0.299R + 0.587G + 0.114B
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness

def adaptive_shadow_residual(shadow_img, non_shadow_img, threshold=0.1):
    """
    计算自适应阴影残差
    输入: shadow_img, non_shadow_img - torch tensor [3, H, W] 范围 0-1
    """
    shadow_brightness = rgb_to_brightness(shadow_img)
    non_shadow_brightness = rgb_to_brightness(non_shadow_img)
    
    # 相对差异，考虑图像的整体亮度水平
    relative_diff = (non_shadow_brightness - shadow_brightness) / (non_shadow_brightness + 1e-8)
    residuals = torch.clamp(relative_diff, min=0)
    
    # 可选：应用阈值去除微小差异
    residuals = torch.where(residuals > threshold, residuals, 0.0)
    
    return residuals

def calculate_mask(residuals):
    residuals = residuals.squeeze()

    median_residual = torch.median(residuals)
    inlier_loss = torch.where(residuals <= median_residual, 1.0, 0.0)

    kernel = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()
    has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
    has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, kernel, padding = "same")
    has_inlier_neighbors = torch.where(has_inlier_neighbors >= 0.5, 1.0, 0.0)

    kernel_16 = 1/(16*16) * torch.ones((1,1,16,16)).cuda()
    if has_inlier_neighbors.shape[1] % 8 != 0:
        pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
    else:
        pad_h = 8

    if has_inlier_neighbors.shape[2] % 8 != 0:
        pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
    else:
        pad_w = 8

    padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
    padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

    is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), kernel_16, stride = 8)

    is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
    is_inlier_patch = is_inlier_patch.squeeze()

    padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

    if padding_indexing[1] == 0:
        padding_indexing[1] = has_inlier_neighbors.shape[1] + padding_indexing[0]
    if padding_indexing[3] == 0:
        padding_indexing[3] = has_inlier_neighbors.shape[2] + padding_indexing[2]

    is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

    is_inlier_patch = torch.where(is_inlier_patch >= 0.6, 1.0, 0.0)

    mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze() >= 1e-3).cuda()

    return mask



class IndoorRobustLoss(torch.nn.Module):
    def __init__(self, n_residuals = 1, hidden_size = 1, per_channel = False):
        super(IndoorRobustLoss, self).__init__()
        # Define a learnable parameter
        self.n_residuals = n_residuals
        self.linear1 = torch.nn.Linear(n_residuals, hidden_size, device = 'cuda:0')
        self.sigmoid1 = torch.nn.Sigmoid()

        self.per_channel = per_channel
        if per_channel:
            self.channel = 3
        else:
            self.channel = 1

        self.kernel_size = 16
        kernel_16 = 1/(self.kernel_size*self.kernel_size) * torch.ones((self.kernel_size, self.kernel_size))
        self.kernel_16 = kernel_16.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channel, self.channel, 1, 1).cuda()
        kernel_3 = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
        self.kernel_3 = kernel_3.view(1, 1, 3, 3).repeat(self.channel, self.channel, 1, 1).cuda()

    def forward(self, residuals):
        medians = torch.median(residuals[0].flatten(start_dim=1), dim=1)[0]

        if self.per_channel == True:
            for i in range(self.channel):
                residuals[0, i] = residuals[0, i] - medians[i]
            inlier_loss = residuals[0]
        else:
            inlier_loss = residuals[0] - torch.median(residuals[0].flatten())      
                
        has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)

        has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, self.kernel_3, padding = "same")

        if has_inlier_neighbors.shape[2] % 8 != 0:
            pad_h = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
        else:
            pad_h = 8

        if has_inlier_neighbors.shape[3] % 8 != 0:
            pad_w = 8 - (has_inlier_neighbors.shape[3] % 8) + 8
        else:
            pad_w = 8

        padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))

        padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

        is_inlier_patch = torch.nn.functional.conv2d(padded_weights.squeeze(0), self.kernel_16, stride = 8)

        is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch.unsqueeze(0), scale_factor = (8,8))

        is_inlier_patch = is_inlier_patch.squeeze()

        padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

        if padding_indexing[1] == 0:
            padding_indexing[1] = has_inlier_neighbors.shape[2] + padding_indexing[0]
        if padding_indexing[3] == 0:
            padding_indexing[3] = has_inlier_neighbors.shape[3] + padding_indexing[2]

        if self.per_channel == True:
            is_inlier_patch = is_inlier_patch[:, padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]
        else:
            is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]
            is_inlier_patch = is_inlier_patch.unsqueeze(0)

        mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze()).cuda()
        mask_before_log = mask

        shape = mask.shape
        res = torch.flatten(residuals, 1)
        res = res[1:]
        res = torch.transpose(res, 0, 1)

        mask = mask.flatten().unsqueeze(1)
        if self.n_residuals > 1:
            mask = torch.cat((res, mask), 1)
        mask = self.linear1(mask)
        mask = self.sigmoid1(mask)
        mask = mask.reshape(shape)
        if self.per_channel == True:
            mask = torch.median(mask, dim=0, keepdim=True)[0]
        else:
            mask = mask.unsqueeze(0)

        return mask, mask_before_log
    
    def threshold(self, linear, sigmoid, x, residuals):
        shape = x.shape
        res = torch.flatten(residuals, 1)
        res = torch.transpose(res, 0, 1)

        x = x.flatten().unsqueeze(1)

        res = torch.cat((res, x), 1)
        x = linear(res)
        x = sigmoid(x)
        x = x.reshape(shape)
        return x
    
    def zero_center(self, x):
        return x - torch.mean(x)
    
"""
shadow_img = read_image_as_tensor('./data/dog/images/0001.png')
non_shadow_img =  read_image_as_tensor('test.png',1080,1920)
residual = adaptive_shadow_residual(shadow_img,non_shadow_img).cuda()
mask = calculate_mask(residual)
save_tensor_as_image(mask,'test1.jpg')

from robust_util import shadow_overlap,process_binary_mask_to_coco
import cv2

shadow_img = cv2.imread('./data/dog/all_shadow_masks/0001.png', cv2.IMREAD_GRAYSCALE)
shadow_coco = process_binary_mask_to_coco(shadow_img)
final_mask = shadow_overlap(~mask,shadow_coco)
final_mask = torch.from_numpy(final_mask)
save_tensor_as_image(final_mask,'test2.jpg')
"""
