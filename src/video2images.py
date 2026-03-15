import cv2
import os
import argparse

def video_to_frames(video_path, output_dir, frame_interval=3):
    """
    将视频转换为帧图片
    
    Args:
        video_path (str): 视频文件路径
        output_dir (str): 输出图片的目录
        frame_interval (int): 帧间隔，默认为1（保存每一帧）
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息:")
    print(f"  - 文件路径: {video_path}")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 帧间隔: {frame_interval}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败，退出循环
        if not ret:
            break
        
        # 按间隔保存帧
        if frame_count % frame_interval == 0:
            # 生成文件名
            filename = f"{saved_count:06d}.jpg"  # 修改为COLMAP期望的格式
            filepath = os.path.join(output_dir, filename)
            
            # 保存帧为图片
            success = cv2.imwrite(filepath, frame)
            if success:
                saved_count += 1
            else:
                print(f"警告：无法保存图片 {filename}")
            
            # 显示进度
            if saved_count % 100 == 0:
                print(f"已保存 {saved_count} 张图片...")
        if frame_count % frame_interval == 1:
            # 生成文件名
            filename = f"{saved_count:06d}.jpg"  # 修改为COLMAP期望的格式
            filepath = os.path.join(output_dir, filename)
            
            # 保存帧为图片
            success = cv2.imwrite(filepath, frame)
            if success:
                saved_count += 1
            else:
                print(f"警告：无法保存图片 {filename}")
            
            # 显示进度
            if saved_count % 100 == 0:
                print(f"已保存 {saved_count} 张图片...")
        frame_count += 1
    
    # 释放视频捕获对象
    cap.release()
    
    print(f"\n转换完成！")
    print(f"总共处理了 {frame_count} 帧")
    print(f"成功保存了 {saved_count} 张图片到 '{output_dir}' 目录")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='将MP4视频转换为图片帧')
    parser.add_argument('--video_path', '-v', required=True, 
                       help='输入视频文件的路径')
    parser.add_argument('--output_dir', '-o', required=True,
                       help='输出目录路径')
    parser.add_argument('--frame_interval', '-i', type=int, default=3,
                       help='帧间隔，每N帧保存一帧 (默认: 1)')
    
    args = parser.parse_args()
    
    # 调用转换函数
    success = video_to_frames(args.video_path, args.output_dir, args.frame_interval)
    
    if success:
        print(f"\n使用方法提示:")
        print(f"1. 对于3D Gaussian Splatting，确保输出目录包含在data/input/images/中")
        print(f"2. 然后运行: python convert.py -s ./data/input")
    else:
        print("转换失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())