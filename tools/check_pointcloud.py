import numpy as np
import open3d as o3d
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_pointcloud(file_path):
    """
    加载点云文件（支持npy、bin格式）
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.npy':
        points = np.load(file_path)
    elif file_extension == '.bin':
        points = np.fromfile(file_path, dtype=np.float32)
        if points.size % 5 == 0:
            points = points.reshape(-1, 5)
        else:
            points = points.reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
        
    return points

def analyze_pointcloud(points):
    """
    分析点云数据的基本特征
    """
    info = {
        "点数量": len(points),
        "维度": points.shape[1],
        "数据类型": points.dtype,
        "坐标范围": {
            "X": [points[:, 0].min(), points[:, 0].max()],
            "Y": [points[:, 1].min(), points[:, 1].max()],
            "Z": [points[:, 2].min(), points[:, 2].max()]
        },
        "是否有NaN": np.any(np.isnan(points)),
        "是否有Inf": np.any(np.isinf(points))
    }
    
    if points.shape[1] >= 4:
        info["强度范围"] = [points[:, 3].min(), points[:, 3].max()]
    
    return info

def visualize_comparison(npy_points, bin_points, title="Point Cloud Comparison"):
    """
    可视化对比两个点云数据
    """
    fig = plt.figure(figsize=(20, 10))
    
    # 2D比较（俯视图）
    ax1 = fig.add_subplot(121)
    ax1.scatter(npy_points[:, 0], npy_points[:, 1], c='blue', s=1, alpha=0.5, label='NPY')
    ax1.scatter(bin_points[:, 0], bin_points[:, 1], c='red', s=1, alpha=0.5, label='BIN')
    ax1.set_title("2D Comparison (Top View)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 3D比较
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(npy_points[:, 0], npy_points[:, 1], npy_points[:, 2], 
                c='blue', s=1, alpha=0.5, label='NPY')
    ax2.scatter(bin_points[:, 0], bin_points[:, 1], bin_points[:, 2], 
                c='red', s=1, alpha=0.5, label='BIN')
    ax2.set_title("3D Comparison")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def compare_files(npy_file, bin_file):
    """
    比较NPY和BIN文件的点云数据
    """
    print(f"比较文件:")
    print(f"NPY: {npy_file}")
    print(f"BIN: {bin_file}")
    print("-" * 50)
    
    # 加载数据
    npy_points = load_pointcloud(npy_file)
    bin_points = load_pointcloud(bin_file)
    
    # 分析NPY数据
    print("\nNPY文件分析:")
    npy_info = analyze_pointcloud(npy_points)
    for key, value in npy_info.items():
        print(f"{key}: {value}")
    
    # 分析BIN数据
    print("\nBIN文件分析:")
    bin_info = analyze_pointcloud(bin_points)
    for key, value in bin_info.items():
        print(f"{key}: {value}")
    
    # 比较数据
    if npy_points.shape == bin_points.shape:
        print("\n数据比较:")
        diff = np.abs(npy_points - bin_points)
        print(f"最大差异: {diff.max()}")
        print(f"平均差异: {diff.mean()}")
        print(f"是否完全相同: {np.allclose(npy_points, bin_points)}")
        
        # 可视化比较
        fig = visualize_comparison(npy_points, bin_points)
        plt.show()
    else:
        print("\n警告: 数据形状不同，无法直接比较")
        print(f"NPY形状: {npy_points.shape}")
        print(f"BIN形状: {bin_points.shape}")

def main():
    parser = argparse.ArgumentParser(description="Point Cloud File Comparison Tool")
    parser.add_argument("--npy", required=True, help="Path to NPY file")
    parser.add_argument("--bin", required=True, help="Path to BIN file")
    args = parser.parse_args()
    
    if not os.path.exists(args.npy) or not os.path.exists(args.bin):
        print("Error: One or both input files do not exist!")
        return
    
    compare_files(args.npy, args.bin)

if __name__ == "__main__":
    # 示例用法
    npy_file = "data/custom_zzg_0325_2class_11846_1500/points_test/npy/1732700639.792158.npy"
    bin_file = "data/custom_zzg_0325_2class_11846_1500/points_test/bin/1732700639.792158.bin"
    compare_files(npy_file, bin_file)
    
    # 也可以通过命令行使用：
    # python tools/check_pointcloud.py --npy path/to/file.npy --bin path/to/file.bin 