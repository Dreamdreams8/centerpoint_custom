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
                       boxes[i,3], boxes[i,4], -boxes[i,8], color)
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

def main():
    # 加载配置
    cfg = Config.fromfile('configs/custom/custom_centerpoint_voxelnet.py')
    
    # 构建模型
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    # 构建数据集
    dataset = build_dataset(cfg.data.val)
    # print("dataset: ",cfg.data.val)
    # print("dataset: ",cfg.data.train)
    
    # dataset = build_dataset(cfg.data.train)
    # 构建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )
    
    # 加载checkpoint
    checkpoint = load_checkpoint(model, 'work_dirs/custom_centerpoint_voxelnet_0418/epoch_26.pth', map_location="cpu")
    model.eval()
    model = model.cuda()
    
    cpu_device = torch.device("cpu")
    
    # 创建输出目录
    out_dir = "data/custom_0411_zzg/visualization_0418_26epoch"
    pred_dir = os.path.join(out_dir, "pred")  # 预测结果目录
    gt_dir = os.path.join(out_dir, "gt")      # 标签结果目录
    
    for d in [pred_dir, gt_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 获取点云范围用于可视化
    point_cloud_range = [-69.12, -39.68, -1, 69.12, 39.68, 7]
    
    # 处理每个样本
    for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # 获取点云数据
        points = data_batch['points'][0][:, 0:3].cpu().numpy()
        
        # ['metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates']
        # 获取点云文件名
        lidar_token = data_batch['metadata'][0]['token']
        
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
        pred_save_path = os.path.join(pred_dir, f"{lidar_token}.png")
        visualize_result(points.T, outputs[0], pred_save_path, point_cloud_range)
        

if __name__ == '__main__':
    main()

# python tools/demo_result_show_2d.py 