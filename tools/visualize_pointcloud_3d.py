import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import threading
import queue

class VisualizerWithKeyCallback:
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Point Cloud Viewer", width=1600, height=900)
        
        self.is_run = True
        self.next_frame = False
        
        # 创建一个空的点云对象并添加到场景中
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        # 注册按键回调
        self.vis.register_key_callback(32, self.space_callback)    # 空格键
        self.vis.register_key_callback(ord('Q'), self.quit_callback)
        self.vis.register_key_callback(ord('R'), self.reset_view_callback)  # R键重置视角
        
    def reset_view(self):
        """重置视角"""
        try:
            ctr = self.vis.get_view_control()
            if ctr is not None:
                ctr.set_zoom(0.3)
                ctr.set_front([-0.5, -0.5, -0.5])
                ctr.set_lookat([0, 0, 0])
                ctr.set_up([0, 0, 1])
        except:
            print("Warning: Failed to set view")
            
    def reset_view_callback(self, vis):
        self.reset_view()
        return False
        
    def space_callback(self, vis):
        self.next_frame = True
        return False
    
    def quit_callback(self, vis):
        self.is_run = False
        return False
    
    def update_geometries(self, points, colors=None):
        """更新点云数据"""
        if len(points) == 0:
            return
            
        # 更新点云数据
        self.pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # 使用默认颜色
            default_colors = np.ones((len(points), 3)) * np.array([1, 0.706, 0])  # 金色
            self.pcd.colors = o3d.utility.Vector3dVector(default_colors)
        
        # 强制更新
        try:
            self.vis.update_geometry(self.pcd)
        except:
            try:
                # 如果update_geometry失败，尝试直接更新渲染器
                self.vis.remove_geometry(self.pcd)
                self.vis.add_geometry(self.pcd)
            except:
                pass
        
        self.vis.poll_events()
        self.vis.update_renderer()

def create_line_set(boxes, scores, labels, score_thr=0.3):
    """创建用于显示3D边界框的线集"""
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],  # 底部四条边
             [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部四条边
             [0, 4], [1, 5], [2, 6], [3, 7]]  # 四个垂直边
    
    colors = [
        [1, 0, 0],  # 红色 - car
        [0, 0, 1],  # 蓝色 - truck
        [0.5, 0.5, 0.5],  # 灰色
        [0, 1, 0],  # 绿色
        [1, 1, 0],  # 黄色
    ]
    
    all_corners = []
    all_lines = []
    all_colors = []
    
    base_lines = np.array(lines)
    
    for i, box in enumerate(boxes):
        if scores[i] < score_thr:
            continue
            
        # 计算8个角点的坐标
        center = box[:3]
        dims = box[3:6]
        yaw = box[6]
        
        # 创建旋转矩阵
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 生成8个角点
        corners = np.array([
            [-dims[0]/2, -dims[1]/2, -dims[2]/2],
            [dims[0]/2, -dims[1]/2, -dims[2]/2],
            [dims[0]/2, dims[1]/2, -dims[2]/2],
            [-dims[0]/2, dims[1]/2, -dims[2]/2],
            [-dims[0]/2, -dims[1]/2, dims[2]/2],
            [dims[0]/2, -dims[1]/2, dims[2]/2],
            [dims[0]/2, dims[1]/2, dims[2]/2],
            [-dims[0]/2, dims[1]/2, dims[2]/2],
        ])
        
        # 应用旋转和平移
        corners = corners @ rot_mat.T + center
        
        # 添加到列表中
        all_corners.extend(corners)
        all_lines.extend(base_lines + len(all_corners) - 8)
        all_colors.extend([colors[labels[i] % len(colors)]] * 12)  # 12条线
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_corners)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(all_colors)
    
    return line_set

def visualize_sequence(points_list=None, point_cloud_range=None, data_dir=None):
    """可视化点云序列"""
    # 创建可视化器
    vis_ctrl = VisualizerWithKeyCallback()
    
    if data_dir is not None:
        # 流式加载模式
        bin_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
        total_frames = len(bin_files)
        
        current_frame = 0
        while current_frame < total_frames and vis_ctrl.is_run:
            try:
                file = bin_files[current_frame]
                
                # 加载当前帧
                points = np.fromfile(os.path.join(data_dir, file), dtype=np.float32)
                if points.size % 5 == 0:
                    points = points.reshape(-1, 5)
                else:
                    points = points.reshape(-1, 4)
                
                # 准备点云数据
                points_xyz = points[:, :3]
                
                # 过滤无效点
                valid_mask = ~np.isnan(points_xyz).any(axis=1)
                points_xyz = points_xyz[valid_mask]
                
                if points.shape[1] >= 4:
                    colors = np.zeros((points_xyz.shape[0], 3))
                    intensities = points[valid_mask, 3]
                    norm_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
                    colors[:, 0] = norm_intensities  # 使用红色通道表示强度
                else:
                    colors = None
                
                # 更新场景
                vis_ctrl.update_geometries(points_xyz, colors)
                
                # 显示当前帧信息
                print(f"\rFrame: {current_frame+1}/{total_frames} | File: {file} | Points: {len(points_xyz)} (Space: next frame, Q: quit, R: reset view)", end="", flush=True)
                
                # 等待空格键或退出信号
                while not vis_ctrl.next_frame and vis_ctrl.is_run:
                    vis_ctrl.vis.poll_events()
                    vis_ctrl.vis.update_renderer()
                    time.sleep(0.01)
                
                if not vis_ctrl.is_run:
                    break
                
                # 重置next_frame标志并前进到下一帧
                vis_ctrl.next_frame = False
                current_frame += 1
                
            except Exception as e:
                print(f"\nError processing file {file}: {str(e)}")
                current_frame += 1
                continue
    
    vis_ctrl.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Point Cloud 3D Visualization")
    parser.add_argument("--input", default="data/custom_zzg_0325_2class_11846_1500/points",
                      help="Input point cloud file or directory")
    parser.add_argument("--range", nargs='+', type=float, 
                       default=[-69.12, 60.12, -1, 69.12, 60.12, 7],
                       help="Point cloud range: xmin ymin zmin xmax ymax zmax")
    args = parser.parse_args()
    args.input = "data/custom_zzg_0325_2class_11846_1500/points_test"
    if os.path.isdir(args.input):
        visualize_sequence(data_dir=args.input, point_cloud_range=args.range)
    else:
        print("Error: Input path does not exist!")

if __name__ == '__main__':
    main()

# 使用示例：
# python tools/visualize_pointcloud_3d.py

# 使用示例：
# python tools/visualize_pointcloud_3d.py \
#     --input data/custom/points/1732700583.420497.bin \
#     --out-dir data/custom/visualization

# 文件夹连续可视化：
# python tools/visualize_pointcloud_3d.py \
#     --input dcustom_zzg_0325_2class_11846_1500 \
#     --out-dir data/custom/visualization \
#     --delay 0.1

# 指定可视化范围：
# python tools/visualize_pointcloud_3d.py \
#     --input data/custom/points \
#     --out-dir data/custom/visualization \
#     --range -50 -50 -5 50 50 3 