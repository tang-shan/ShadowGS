from robust_loss import calculate_mask,adaptive_shadow_residual,save_tensor_as_image
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_image_to_tensor(image_path, normalize=True, dtype=torch.float32):
    """
    读取图片并转换为[3, h, w]的PyTorch Tensor
    
    Args:
        image_path (str): 图片文件路径
        normalize (bool): 是否归一化到[0,1]，如果为False则保持[0,255]范围
        dtype: 输出的数据类型
        
    Returns:
        torch.Tensor: 形状为[3, h, w]的Tensor
    """
    # 使用PIL读取图片
    image = Image.open(image_path)
    
    # 确保是RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor()  # 自动归一化到[0,1]
        ])
    else:
        transform = transforms.Compose([
            transforms.PILToTensor()  # 不归一化，保持[0,255]
        ])
    
    # 应用转换
    tensor = transform(image).to(dtype)
    
    return tensor

def save_grayscale_tensor(tensor, save_path):
    """
    将[h, w]形状的Tensor保存为灰度图
    
    Args:
        tensor (torch.Tensor): 形状为[h, w]的Tensor
        save_path (str): 保存路径
    """
    # 检查输入形状
    if tensor.dim() != 2:
        raise ValueError(f"期望2D Tensor [h, w]，但得到形状: {tensor.shape}")
    
    # 分离计算图，转换为numpy
    tensor_np = tensor.detach().cpu().numpy()
    
    # 归一化到[0, 255]范围
    if tensor_np.max() > 1.0 or tensor_np.min() < 0.0:
        # 如果值范围不在[0,1]，进行归一化
        tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) * 255
    else:
        # 如果已经在[0,1]范围，直接映射到[0,255]
        tensor_np = tensor_np * 255
    
    # 转换为uint8类型
    tensor_np = tensor_np.astype(np.uint8)
    
    # 创建PIL图像并保存
    image = Image.fromarray(tensor_np, mode='L')
    image.save(save_path)
    print(f"灰度图已保存到: {save_path}")

gt_image_path = './data/bench/images/000016.jpg'
image_path = './results/bench/train/ours_30000/renders/00016.png'

gt_image = load_image_to_tensor(gt_image_path)
image = load_image_to_tensor(image_path)

residual = adaptive_shadow_residual(gt_image,image)
save_grayscale_tensor(residual,'tmp.jpg')
