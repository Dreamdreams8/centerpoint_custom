import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

def visualize_points(points, save_path, point_cloud_range=None):
    """可视化点云并保存"""
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    
    if point_cloud_range is None:
        # 如果没有指定范围，使用点云的实际范围
        margin = 5  # 添加一些边距
        xlim = [points[0].min() - margin, points[0].max() + margin]
        ylim = [points[1].min() - margin, points[1].max() + margin]
    else:
        xlim = [point_cloud_range[0], point_cloud_range[3]]
        ylim = [point_cloud_range[1], point_cloud_range[4]]
    
    # 设置坐标范围
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    
    # 绘制点云
    ax.scatter(points[0], points[1], s=1, c='white')
    
    # 保存图片
    plt.axis('off')
    plt.savefig(save_path, dpi=50, bbox_inches='tight', pad_inches=0, 
                facecolor='black', format='png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Visualization")
    parser.add_argument("--input", help="Input point cloud file or directory", required=True)
    parser.add_argument("--out-dir", help="Output directory for visualization", required=True)
    parser.add_argument("--range", nargs='+', type=float, 
                       default=[-69.12, -39.68, -1, 69.12, 39.68, 7],
                       help="Point cloud range: xmin ymin zmin xmax ymax zmax")
    args = parser.parse_args()
    
    # 创建输出目录
    out_dir = os.path.join(args.out_dir, "points")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 单文件模式
        points = np.fromfile(args.input, dtype=np.float32)
        if points.size % 5 == 0:
            points = points.reshape(-1, 5)[:, :3]
        else:
            points = points.reshape(-1, 4)[:, :3]
        
        save_path = os.path.join(out_dir, 
                               os.path.splitext(os.path.basename(args.input))[0] + '.png')
        visualize_points(points.T, save_path, args.range)
        print(f"Visualization saved to {save_path}")

    elif os.path.isdir(args.input):
        # 文件夹模式
        bin_files = sorted([f for f in os.listdir(args.input) if f.endswith('.bin')])
        
        for file in tqdm(bin_files, desc="Processing"):
            points = np.fromfile(os.path.join(args.input, file), dtype=np.float32)
            if points.size % 5 == 0:
                points = points.reshape(-1, 5)[:, :3]
            else:
                points = points.reshape(-1, 4)[:, :3]
            
            save_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.png')
            visualize_points(points.T, save_path, args.range)
            
        print(f"All visualizations saved to {out_dir}")
    else:
        print("Error: Input path does not exist!")

if __name__ == '__main__':
    main()

# 使用示例：
# 处理单个文件：
# python tools/visualize_pointcloud.py \
#     --input data/custom_zzg_0325_2class_11846_1500/points/1732700639.792158.bin \
#     --out-dir data/custom_zzg_0325_2class_11846_1500/visualization

# 处理文件夹：
# python tools/visualize_pointcloud.py \
#     --input data/custom_zzg_0325_2class_11846_1500/points \
# --out-dir data/custom_zzg_0325_2class_11846_1500/visualization

# 指定可视化范围：
# python tools/visualize_pointcloud.py \
#     --input data/custom/points \
#     --out-dir data/custom/visualization \
#     --range -50 -50 -5 50 50 3 