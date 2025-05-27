import argparse
import copy
import json
import os
import sys
from tqdm import tqdm
try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 从result_show_2d.py引入的可视化相关代码
box_colormap = [
    [1, 0, 0],     # 红色 - car
    [0, 0, 1],     # 蓝色 - truck
    [0.5, 0.5, 0.5], # 灰色
    [0, 1, 0],     # 绿色
    [1, 1, 0],     # 黄色
    [0, 1, 1],     # 青色
    [1, 0, 1],     # 紫色
    [0.5, 0, 0],   # 深红色
    [0, 0.5, 0],   # 深绿色
    [0, 0, 0.5],   # 深蓝色
]

def draw_2d_box(ax, x, y, w, h, theta_rad, color='r'):
    """绘制2D边界框"""
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad), np.cos(theta_rad)]])
    
    corners = np.array([[-w/2, -h/2],
                       [w/2, -h/2],
                       [w/2, h/2],
                       [-w/2, h/2]])
    
    rotated_corners = np.dot(corners, R.T)
    rotated_corners += np.array([x, y])
    
    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=2)
    ax.plot([rotated_corners[0, 0], rotated_corners[-1, 0]],
            [rotated_corners[0, 1], rotated_corners[-1, 1]], color=color, linewidth=2)
    # Add heading direction indicator (front of the vehicle)
    front_point = np.array([[w/2, 0]])  # Front center point in local coordinates
    rotated_front = np.dot(front_point, R.T) + np.array([x, y])
    ax.plot([x, rotated_front[0, 0]], 
            [y, rotated_front[0, 1]], 
            color=color, linewidth=2, linestyle='--')    

def visualize_result(points, detection, save_path, point_cloud_range):
    """可视化点云和检测结果并保存"""
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    
    # 设置坐标范围
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    
    # 绘制点云
    ax.scatter(points[0], points[1], s=1, c='white')
    
    # 绘制检测框
    boxes = detection['box3d_lidar']
    scores = detection['scores']
    labels = detection['label_preds']
    for i in range(len(boxes)):
        if scores[i] > 0.3:
            color = box_colormap[labels[i]]
            draw_2d_box(ax, boxes[i,0], boxes[i,1], 
                       boxes[i,3], boxes[i,4], boxes[i,8], color)
            # draw_2d_box(ax, boxes[i,0], boxes[i,1], 
            #            boxes[i,3], boxes[i,4], -boxes[i,6], color)            
            # draw_2d_box(ax, boxes[i,1], -boxes[i,0], 
            #            boxes[i,4], boxes[i,3], boxes[i,8], color)   
    # 保存图片
    plt.axis('off')
    plt.savefig(save_path, dpi=50, bbox_inches='tight', pad_inches=0, 
                facecolor='black', format='png')
    plt.close()

def convert_box(info):
    """从demo.py复制的转换函数"""
    boxes = info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]
    
    assert len(boxes) == len(names)
    
    detection = {}
    detection['box3d_lidar'] = boxes
    detection['label_preds'] = np.zeros(len(boxes)) 
    detection['scores'] = np.ones(len(boxes))
    
    return detection

def read_bin_file(file_path):
    # 假设点云格式为 (x, y, z, intensity)，shape=(N, 4)
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    zeros_column = np.zeros((points.shape[0], 1), dtype=points.dtype)
    points = np.hstack((points, zeros_column))  
    return points
    # return points[:, :3]  

from det3d.core.input.voxel_generator import VoxelGenerator
def create_data_batch(cfg,points):

    # voxel_generator = dict(
    #     range=[-54, -54, -1.0, 54, 54, 7.0],
    #     voxel_size=[0.075, 0.075, 0.2],
    #     max_points_in_voxel=10,
    #     max_voxel_num=[120000, 160000],
    # )    

    range = [-54, -54, -1.0, 54, 54, 7.0]
    voxel_size = [0.075, 0.075, 0.2]
    max_points_in_voxel = 10
    max_voxel_num = [120000, 160000]
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=range,
        max_num_points=max_points_in_voxel,
        max_voxels=max_voxel_num[0],
    )
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size    

    max_voxels = max_voxel_num[1]
    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels=max_voxels 
    )
    zeros_column = np.zeros((coordinates.shape[0], 1), dtype=points.dtype)
    coordinates = np.hstack((zeros_column,coordinates))     
    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
    # -voxel num:  [45877]
    # -coordinates:  (45877, 3)
    # -num_points:  (45877,)
    # -voxels:   (45877, 10, 5)    


    data_batch = {
        "points": [torch.tensor( points, dtype=torch.float32).cuda()],  # batch_size=1
        "voxels": torch.tensor( voxels, dtype=torch.float32).cuda(),
        "coordinates": torch.tensor( coordinates, dtype=torch.int32).cuda(),
        "num_points": torch.tensor( num_points, dtype=torch.int32).cuda(),
        "num_voxels": torch.tensor( num_voxels, dtype=torch.int64),
        "shape": [[1440,1440,40]],  # 保持为 list
        "metadata": [{"token": "custom_frame"}]
    }
 
    return data_batch

def main():
    # 加载配置
    # cfg = Config.fromfile('configs/custom/custom_centerpoint_voxelnet.py')
    cfg = Config.fromfile('checkpoint/custom_centerpoint_voxelnet_0522_9class.py')
    
    # 构建模型
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    # 加载checkpoint
    checkpoint = load_checkpoint(model, 'checkpoint/epoch_400_0522_9class.pth', map_location="cpu")
    model.eval()
    model = model.cuda()
    
    cpu_device = torch.device("cpu")
    bin_folder = "data/custom/points"
    bin_lists = os.listdir(bin_folder)
    sorted_bin_lists = sorted(bin_lists, key=lambda x: float(os.path.splitext(x)[0]))
    # 创建输出目录
    out_dir = "data/custom_0411_zzg/visualization_0527_test"
    pred_dir = os.path.join(out_dir, "pred")  # 预测结果目录
    gt_dir = os.path.join(out_dir, "gt")      # 标签结果目录
    
    for d in [pred_dir, gt_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 获取点云范围用于可视化
    point_cloud_range = [-54, -54, -1.0, 54, 54, 7.0]
    for file in tqdm(sorted_bin_lists):
        bin_path = os.path.join(bin_folder, file)
        points = read_bin_file(bin_path)
        # 生成 data_batch
        data_batch = create_data_batch(cfg,points)
        # 模型推理
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )     
        # 处理输出结果
        for output in outputs:
            for k, v in output.items():
                if k not in ["metadata"]:
                    output[k] = v.to(cpu_device)
        
        # 保存预测结果可视化
        # pred_save_path = os.path.join(pred_dir, "point_cloud.png")
        pred_save_path = os.path.join(pred_dir, f"{file}.png")
        visualize_result(points.T, outputs[0], pred_save_path, point_cloud_range)
                        

if __name__ == '__main__':
    main()

# python tools/demo_result_show_2d.py 