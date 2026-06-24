#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import matplotlib.pyplot as plt
import numpy as np



def calculate_brightness_diff(img1, img2, grayscale=False):
    """
    计算两个图像的亮度差异
    
    Args:
        img1: torch.Tensor [3, h, w]
        img2: torch.Tensor [3, h, w]
        grayscale: 是否转换为灰度计算
    
    Returns:
        diff: 亮度差异图
    """
    if grayscale:
        # 灰度计算
        weights = torch.tensor([0.299, 0.587, 0.114], device=img1.device).view(3, 1, 1)
        gray1 = (img1 * weights).sum(dim=0)
        gray2 = (img2 * weights).sum(dim=0)
        brightness_diff = gray1 - gray2
    else:
        # 彩色计算
        brightness_diff = img1 - img2
    
    # 将小于0的值设为0
    brightness_diff = torch.clamp(brightness_diff, min=0)
    
    return brightness_diff

def visualize_and_save_brightness_diff(img1,img2,brightness_diff, save_path='test.jpg'):
    """
    计算亮度差异并可视化保存
    
    Args:
        img1: torch.Tensor [3, h, w]
        img2: torch.Tensor [3, h, w]
        save_path: 保存路径
    """
    
    # 转换为numpy用于可视化
    diff_np = brightness_diff.detach().cpu().numpy()
    
    # 创建可视化图像
    plt.figure(figsize=(12, 4))
    
    # 子图1: 原图1
    plt.subplot(1, 3, 1)
    img1_np = img1.permute(1, 2, 0).detach().cpu().numpy()
    if img1_np.shape[2] == 3:  # RGB图像
        plt.imshow(np.clip(img1_np, 0, 1))  # 假设值在[0,1]范围内
    else:
        plt.imshow(img1_np, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    # 子图2: 原图2
    plt.subplot(1, 3, 2)
    img2_np = img2.permute(1, 2, 0).detach().cpu().numpy()
    if img2_np.shape[2] == 3:  # RGB图像
        plt.imshow(np.clip(img2_np, 0, 1))
    else:
        plt.imshow(img2_np, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    # 子图3: 亮度差异
    plt.subplot(1, 3, 3)
    im = plt.imshow(diff_np, cmap='hot', vmin=0, vmax=diff_np.max() if diff_np.max() > 0 else 1)
    plt.title('Brightness Difference\n(img1 - img2, <0 → 0)')
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"亮度差异图已保存为: {save_path}")
    print(f"亮度差异统计: min={diff_np.min():.4f}, max={diff_np.max():.4f}, mean={diff_np.mean():.4f}")
    
    return brightness_diff

def binary_threshold_adaptive(diff, factor=1.0):
    """
    使用自适应阈值转为二值图
    
    Args:
        diff: 亮度差异图 [h, w]
        factor: 阈值系数，threshold = mean * factor
    """
    threshold = diff.mean() * factor
    binary = (diff > threshold).float()
    return binary

from scipy import ndimage

def remove_small_areas(binary_diff, min_area=50):
    """
    移除二值图中面积很小的区域
    
    Args:
        binary_diff: 二值图 tensor [h, w]
        min_area: 最小面积阈值，小于这个面积的区域会被移除
    
    Returns:
        filtered_binary: 过滤后的二值图
    """
    # 转换为numpy数组
    binary_np = binary_diff.detach().cpu().numpy()
    
    # 进行连通组件分析
    labeled_array, num_features = ndimage.label(binary_np)
    
    # 计算每个连通区域的面积
    areas = ndimage.sum(binary_np, labeled_array, range(1, num_features + 1))
    
    # 创建掩码，只保留面积大于阈值的区域
    mask = np.zeros_like(binary_np, dtype=bool)
    for i in range(num_features):
        if areas[i] >= min_area:
            mask[labeled_array == i + 1] = True
    
    # 转换回tensor
    filtered_binary = torch.from_numpy(mask.astype(np.float32)).to(binary_diff.device)
    
    print(f"原始区域数量: {num_features}")
    print(f"过滤后区域数量: {len([a for a in areas if a >= min_area])}")
    print(f"移除的区域数量: {len([a for a in areas if a < min_area])}")
    
    return filtered_binary


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "shadow_masks")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        gt = view.original_image[0:3, :, :]
        """
        diff = calculate_brightness_diff(rendering,gt,grayscale=True)
        diff[diff<0.2]=0.0
        diff = diff/depth.squeeze()
        """

        diff = calculate_brightness_diff(rendering, gt, grayscale=True)

        # brightness threshold
        diff[diff < 0.2] = 0.0

        # depth threshold t
        t = 10000.0

        depth_map = depth.squeeze()

        # 先除深度
        diff = diff / (depth_map + 1e-6)

        


        diff = binary_threshold_adaptive(diff)
        diff = remove_small_areas(diff)

        # 再过滤近距离区域
        diff[depth_map > t] = 0.0
        binary_np = diff.detach().cpu().numpy()
        
        plt.imsave(os.path.join(mask_path, view.image_name), binary_np, cmap='gray', vmin=0, vmax=1)
        #plt.imsave('binary_map.jpg', binary_np, cmap='gray', vmin=0, vmax=1)

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_stack = []
        val_stack = []
        viewpoint_stack = scene.getTrainCameras()
        for view in viewpoint_stack:
            name = view.image_name.split('.')[0]
            if 'test_' not in name:
                train_stack.append(view)
            else:
                val_stack.append(view)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, train_stack, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, val_stack, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)