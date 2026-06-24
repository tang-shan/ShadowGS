import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from sgs.utils.loss_utils import ssim
from compare.gaussiansplatting.lpipsPyTorch import lpips
import json
from tqdm import tqdm
from sgs.utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
import glob
from torchvision.utils import save_image


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def readImage(path):
    img = Image.open(path)
    return tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()

def read_image_to_binary_tensor(path, threshold=128, target_size=None):
    """
    将图片读取为二值图像的tensor
    
    Args:
        path: 图片路径
        threshold: 二值化阈值 (0-255)，默认128
        target_size: 目标大小 (h, w)，可选
        
    Returns:
        binary_tensor: 二值tensor [h, w]，值为0或1
    """
    # 读取图片
    image = Image.open(path)
    
    # 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
    
    # 调整大小（如果指定了目标大小）
    if target_size is not None:
        h, w = target_size
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 二值化处理
    binary_array = (img_array > threshold).astype(np.float32)
    
    # 转换为tensor
    binary_tensor = torch.from_numpy(binary_array)
    
    return binary_tensor.cuda()

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


def cal_lpips(gt_path,result_path,mask_path=None):
    gt_img = readImage(gt_path)
    result_img = readImage(result_path,target_size=target_size)
    if mask_path is not None:
        target_size = (gt_img.shape[2], gt_img.shape[3])  # H, W
        mask_img = read_image_to_binary_tensor(
                mask_path,
                threshold=128,
                target_size=target_size
            )
        gt_img = gt_img*mask_img
        result_img = result_img*mask_img
    return lpips(result_img,gt_img,net_type='vgg')

def cal_ssim(gt_path,result_path,mask_path=None):
    gt_img = readImage(gt_path)
    result_img = readImage(result_path,target_size=target_size)
    if mask_path is not None:
        target_size = (gt_img.shape[2], gt_img.shape[3])  # H, W
        mask_img = read_image_to_binary_tensor(
                mask_path,
                threshold=128,
                target_size=target_size
            )
        gt_img = gt_img*mask_img
        result_img = result_img*mask_img
    return ssim(result_img,gt_img)


def cal_psnr(gt_path,result_path,mask_path=None):
    gt_img = readImage(gt_path)
    result_img = readImage(result_path,target_size=target_size)
    if mask_path is not None:
        target_size = (gt_img.shape[2], gt_img.shape[3])  # H, W
        mask_img = read_image_to_binary_tensor(
                mask_path,
                threshold=128,
                target_size=target_size
            )
        gt_img = gt_img*mask_img
        result_img = result_img*mask_img
    return psnr(result_img,gt_img)

    

