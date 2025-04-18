import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import argparse
from pathlib import Path
import time

def load_point_cloud(file_path):
    """
    加载点云文件，支持npy、bin和pcd格式
    
    参数:
    file_path: 点云文件路径
    
    返回:
    points: numpy数组，形状为(N, 3)或(N, 4)，包含点云坐标和可选的强度信息
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.npy':
        points = np.load(file_path)
        print("npy文件的形状为：", points.shape)
        print("npy文件的形状为：", points.size)
    elif file_extension == '.bin':
        points = np.fromfile(file_path, dtype=np.float32)
        print("bin文件的形状为：", points.shape)
        print("bin文件的形状为：", points.size)
        points = points.reshape(-1, 4)
        # if points.size % 5 == 0:
        #     points = points.reshape(-1, 5)
        # else:
        #     points = points.reshape(-1, 4)
    elif file_extension == '.pcd':
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # 将颜色信息转换为强度
            intensities = np.mean(colors, axis=1)
            points = np.column_stack([points, intensities])
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return points

def visualize_matplotlib(points, point_cloud_range=None, save_path=None, white_background=True):
    """使用Matplotlib进行2D/3D可视化"""
    # 创建2x1的子图布局
    fig = plt.figure(figsize=(20, 10))
    
    # 设置背景颜色
    bg_color = 'white' if white_background else 'black'
    text_color = 'black' if white_background else 'white'
    point_color = 'black' if white_background else 'white'
    grid_color = 'gray'
    fig.patch.set_facecolor(bg_color)
    
    # 2D俯视图 (左图)
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor(bg_color)
    
    # 设置2D显示范围
    if point_cloud_range is None:
        margin = 5
        xlim = [points[:, 0].min() - margin, points[:, 0].max() + margin]
        ylim = [points[:, 1].min() - margin, points[:, 1].max() + margin]
    else:
        xlim = [point_cloud_range[0], point_cloud_range[3]]
        ylim = [point_cloud_range[1], point_cloud_range[4]]
    
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    
    # 绘制2D点云俯视图
    if points.shape[1] >= 4:  # 如果有强度信息
        intensities = points[:, 3]
        norm_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], 
                             c=norm_intensities,
                             cmap='hot',  # 使用热力图配色
                             s=1,         # 点的大小
                             alpha=0.6)   # 透明度
        plt.colorbar(scatter1, ax=ax1, label='Intensity')
    else:
        ax1.scatter(points[:, 0], points[:, 1], 
                   c=point_color,
                   s=1,
                   alpha=0.6)
    
    # 设置2D图的属性
    ax1.set_aspect('equal')
    ax1.grid(True, color=grid_color, alpha=0.3)
    ax1.set_title('2D Bird\'s Eye View\n' + f'Total points: {len(points)}', color=text_color)
    ax1.set_xlabel('X (m)', color=text_color)
    ax1.set_ylabel('Y (m)', color=text_color)
    ax1.tick_params(colors=text_color)
    
    # 3D视图 (右图)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor(bg_color)
    
    # 绘制3D点云
    if points.shape[1] >= 4:
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=norm_intensities,
                             cmap='hot',
                             s=1,
                             alpha=0.6)
        plt.colorbar(scatter2, ax=ax2, label='Intensity')
    else:
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=point_color,
                   s=1,
                   alpha=0.6)
    
    # 设置3D图的属性
    if point_cloud_range is not None:
        ax2.set_xlim(point_cloud_range[0], point_cloud_range[3])
        ax2.set_ylim(point_cloud_range[1], point_cloud_range[4])
        ax2.set_zlim(point_cloud_range[2], point_cloud_range[5])
    
    ax2.set_title('3D View', color=text_color)
    ax2.set_xlabel('X (m)', color=text_color)
    ax2.set_ylabel('Y (m)', color=text_color)
    ax2.set_zlabel('Z (m)', color=text_color)
    ax2.tick_params(colors=text_color)
    
    # 调整3D视角
    ax2.view_init(elev=20, azim=45)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    if save_path:
        # 保存图片
        plt.savefig(save_path, 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor=bg_color,
                   edgecolor='none')
        plt.close()
        print(f"图片已保存至: {save_path}")
    else:
        # 显示图片
        plt.show()

class Open3DVisualizer:
    def __init__(self, white_background=True):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Point Cloud Viewer", width=1600, height=900)
        
        # 设置背景颜色
        opt = self.vis.get_render_option()
        if opt is not None:
            try:
                bg_color = [1, 1, 1] if white_background else [0, 0, 0]
                opt.background_color = np.asarray(bg_color)
                opt.point_size = 2.0
            except:
                print("Warning: Failed to set render options")
        
        self.is_running = True
        self.vis.register_key_callback(ord('Q'), self.quit_callback)
        self.vis.register_key_callback(ord('R'), self.reset_view_callback)
    
    def quit_callback(self, vis):
        self.is_running = False
        return False
    
    def reset_view_callback(self, vis):
        ctr = vis.get_view_control()
        if ctr is not None:
            ctr.set_zoom(0.3)
            ctr.set_front([-0.5, -0.5, -0.5])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])
        return False
    
    def visualize(self, points):
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 设置点云颜色
        if points.shape[1] >= 4:
            colors = np.zeros((len(points), 3))
            intensities = points[:, 3]
            norm_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            colors[:, 0] = norm_intensities  # 使用红色通道表示强度
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 显示点云
        self.vis.add_geometry(pcd)
        
        print("\nControls:")
        print("Left click + drag: Rotate")
        print("Right click + drag: Pan")
        print("Mouse wheel: Zoom")
        print("R: Reset view")
        print("Q: Quit")
        
        while self.is_running:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)
        
        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Visualization")
    parser.add_argument("--input", 
                       default="data/custom_zzg_0325_2class_11846_1500/points_test/bin/1732700639.792158.bin",
                       help="Input point cloud file (supports .npy, .bin, .pcd)")
    parser.add_argument("--save", 
                       help="Save path for visualization (only for matplotlib mode)")
    parser.add_argument("--range", nargs='+', type=float,
                       default=[-69.12, -39.68, -1, 69.12, 39.68, 7],
                       help="Point cloud range: xmin ymin zmin xmax ymax zmax")
    parser.add_argument("--black-bg", action="store_true",
                       help="Use black background instead of white")
    parser.add_argument("--mode", choices=['matplotlib', 'open3d'], default='open3d',
                       help="Visualization mode: matplotlib for static view, open3d for interactive 3D")
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    # 加载点云数据
    try:
        points = load_point_cloud(args.input)
        print(f"Loaded point cloud with {len(points)} points")
    except Exception as e:
        print(f"Error loading point cloud: {str(e)}")
        return
    
    # 选择可视化方式
    if args.mode == 'matplotlib':
        visualize_matplotlib(points, args.range, args.save, white_background=not args.black_bg)
    else:  # open3d
        visualizer = Open3DVisualizer(white_background=not args.black_bg)
        visualizer.visualize(points)

if __name__ == "__main__":
    main()

# 使用示例：
# Matplotlib模式（静态视图）：
# python tools/singleframe_show.py --mode matplotlib

# Open3D模式（交互式3D视图）：
# python tools/singleframe_show.py --mode open3d

# 支持多种格式：
# python tools/singleframe_show.py --input data/custom_zzg_0325_2class_11846_1500/points_test/npy/1732700639.792158.npy
# python tools/singleframe_show.py --input data/custom_zzg_0325_2class_11846_1500/points_test/bin/1732700700.722201.bin
# python tools/singleframe_show.py --input path/to/file.pcd

# 其他选项：
# --black-bg: 使用黑色背景
# --save: 保存图片（仅Matplotlib模式）
# --range: 指定显示范围 