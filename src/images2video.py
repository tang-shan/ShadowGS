import cv2
import os
import argparse
import glob
import re

def natural_sort_key(filename):
    """
    自然排序键函数，用于正确排序包含数字的文件名
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

def frames_to_video(image_dir, output_path, fps=30):
    """
    将图片帧序列转换为视频
    
    Args:
        image_dir (str): 图片目录路径
        output_path (str): 输出视频文件路径
        fps (int): 输出视频的帧率
    """
    # 检查图片目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误：图片目录 '{image_dir}' 不存在")
        return False
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # 自然排序，确保顺序正确
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if not image_files:
        print(f"错误：在目录 '{image_dir}' 中未找到图片文件")
        return False
    
    print(f"图片信息:")
    print(f"  - 输入目录: {image_dir}")
    print(f"  - 找到图片: {len(image_files)} 张")
    print(f"  - 输出路径: {output_path}")
    print(f"  - 帧率: {fps} FPS")
    
    # 显示前几个文件名用于调试
    print(f"  - 前5个文件: {[os.path.basename(f) for f in image_files[:5]]}")
    
    # 读取第一张图片获取尺寸信息
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误：无法读取第一张图片 '{image_files[0]}'")
        return False
    
    height, width = first_image.shape[:2]
    print(f"  - 图片尺寸: {width} x {height}")
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("错误：无法创建视频文件")
        return False
    
    processed_count = 0
    
    for image_file in image_files:
        # 读取图片
        frame = cv2.imread(image_file)
        
        if frame is None:
            print(f"警告：无法读取图片 '{image_file}'，跳过")
            continue
        
        # 调整图片尺寸以匹配第一张图片（如果需要）
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
            print(f"警告：图片 '{os.path.basename(image_file)}' 尺寸不匹配，已调整")
        
        # 写入视频
        out.write(frame)
        processed_count += 1
        
        # 显示进度
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count} 张图片...")
    
    # 释放视频写入对象
    out.release()
    
    print(f"\n转换完成！")
    print(f"总共处理了 {processed_count} 张图片")
    print(f"视频已保存到: '{output_path}'")
    
    # 计算视频时长
    duration = processed_count / fps
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"视频时长: {duration:.2f} 秒 ({minutes}分{seconds}秒)")
    
    return True


def frames_to_video_pause(image_dir, output_path, pause_seconds=0.1, base_fps=30):
    """
    将图片帧序列转换为视频，每张图片停顿指定时间
    
    Args:
        image_dir (str): 图片目录路径
        output_path (str): 输出视频文件路径
        pause_seconds (float): 每张图片停顿的秒数
        base_fps (int): 输出视频的基础帧率
    """
    # 检查图片目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误：图片目录 '{image_dir}' 不存在")
        return False
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # 自然排序，确保顺序正确
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if not image_files:
        print(f"错误：在目录 '{image_dir}' 中未找到图片文件")
        return False
    
    # 计算每张图片需要重复的帧数
    frames_per_image = max(1, int(pause_seconds * base_fps))
    total_frames = len(image_files) * frames_per_image
    
    print(f"图片信息:")
    print(f"  - 输入目录: {image_dir}")
    print(f"  - 找到图片: {len(image_files)} 张")
    print(f"  - 输出路径: {output_path}")
    print(f"  - 基础帧率: {base_fps} FPS")
    print(f"  - 每张停顿: {pause_seconds} 秒")
    print(f"  - 每张重复: {frames_per_image} 帧")
    print(f"  - 总帧数: {total_frames}")
    
    # 显示前几个文件名用于调试
    print(f"  - 前5个文件: {[os.path.basename(f) for f in image_files[:5]]}")
    
    # 读取第一张图片获取尺寸信息
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误：无法读取第一张图片 '{image_files[0]}'")
        return False
    
    height, width = first_image.shape[:2]
    print(f"  - 图片尺寸: {width} x {height}")
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, base_fps, (width, height))
    
    if not out.isOpened():
        print("错误：无法创建视频文件")
        return False
    
    processed_images = 0
    written_frames = 0
    
    for image_file in image_files:
        # 读取图片
        frame = cv2.imread(image_file)
        
        if frame is None:
            print(f"警告：无法读取图片 '{image_file}'，跳过")
            continue
        
        # 调整图片尺寸以匹配第一张图片（如果需要）
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
            print(f"警告：图片 '{os.path.basename(image_file)}' 尺寸不匹配，已调整")
        
        # 重复写入同一帧以实现停顿效果
        for _ in range(frames_per_image):
            out.write(frame)
            written_frames += 1
        
        processed_images += 1
        
        # 显示进度
        if processed_images % 10 == 0:
            print(f"已处理 {processed_images} 张图片...")
    
    # 释放视频写入对象
    out.release()
    
    print(f"\n转换完成！")
    print(f"总共处理了 {processed_images} 张图片")
    print(f"总共写入 {written_frames} 帧")
    print(f"视频已保存到: '{output_path}'")
    
    # 计算视频时长
    duration = written_frames / base_fps
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"视频时长: {duration:.2f} 秒 ({minutes}分{seconds}秒)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='将图片帧序列转换为MP4视频')
    parser.add_argument('--image_dir', '-i', required=True, 
                       help='输入图片目录路径')
    parser.add_argument('--output_path', '-o', required=True,
                       help='输出视频文件路径 (如: output.mp4)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                       help='输出视频的帧率 (默认: 30)')
    
    args = parser.parse_args()
    
    # 调用转换函数
    #success = frames_to_video_pause(args.image_dir, args.output_path, args.fps)
    success = frames_to_video_pause(args.image_dir, args.output_path)
    
    if not success:
        print("转换失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())