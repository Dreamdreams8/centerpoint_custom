import argparse
import os
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from det3d import torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import batch_processor
from det3d.torchie.trainer import load_checkpoint
from det3d.datasets import build_dataset

# 颜色映射
box_colormap = [
    [1, 0, 0],     # 红色 - car
    [0, 0, 1],     # 蓝色 - truck
    [0.5, 0.5, 0.5], # 灰色
    [0, 1, 0],     # 绿色
    [1, 1, 0],     # 黄色
]

# 标签颜色 - 使用不同颜色区分预测和标签
gt_color = [0, 1, 1]  # 青色

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

def visualize_result(points, detection, save_path, point_cloud_range, gt_boxes=None):
    """可视化点云和检测结果并保存，可选地添加标签框"""
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    
    # 设置坐标范围
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    
    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], s=1, c='white')
    
    # 绘制检测框
    if detection is not None:
        boxes = detection['box3d_lidar']
        scores = detection['scores']
        labels = detection['label_preds']
        
        for i in range(len(boxes)):
            if scores[i] > 0.3:  # 置信度阈值
                color = box_colormap[labels[i]]
                draw_2d_box(ax, boxes[i,0], boxes[i,1], 
                           boxes[i,4], boxes[i,3], -boxes[i,8], color)
    
    # 绘制标签框（如果有）
    if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            draw_2d_box(ax, gt_boxes[i,0], gt_boxes[i,1], 
                       gt_boxes[i,4], gt_boxes[i,3], -gt_boxes[i,8], gt_color)
    
    # 保存图片
    plt.axis('off')
    plt.savefig(save_path, dpi=50, bbox_inches='tight', pad_inches=0, 
                facecolor='black', format='png')
    plt.close()

def load_lidar_file(file_path):
    """加载点云文件"""
    points = np.fromfile(file_path, dtype=np.float32)
    return points.reshape(-1, 4)  # 假设点云格式为 [x, y, z, intensity]

def load_label_file(file_path):
    """加载标签文件"""
    if not os.path.exists(file_path):
        return None
    
    try:
        # 假设标签格式为 [x, y, z, l, w, h, yaw, ...]
        boxes = np.loadtxt(file_path, dtype=np.float32)
        if len(boxes.shape) == 1:
            boxes = boxes.reshape(1, -1)
        return boxes
    except:
        print(f"无法加载标签文件: {file_path}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="点云推理与可视化工具")
    parser.add_argument("--config", default="configs/custom/custom_centerpoint_voxelnet.py", help="配置文件路径")
    parser.add_argument("--checkpoint", default="checkpoint/epoch_22.pth", help="模型检查点路径")
    parser.add_argument("--lidar_dir", default="data/custom_zzg_0325_2class_11846_1500/points", help="点云文件夹路径")
    parser.add_argument("--label_dir", default=None, help="标签文件夹路径（可选）")
    parser.add_argument("--output_dir", default="data/custom_zzg_0325_2class_11846_1500/inference_results", help="输出文件夹路径")
    parser.add_argument("--ext", default=".bin", help="点云文件扩展名")
    parser.add_argument("--label_ext", default=".txt", help="标签文件扩展名")
    parser.add_argument("--threshold", type=float, default=0.3, help="检测置信度阈值")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 构建模型
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    # 加载checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model = model.cuda()
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取点云范围用于可视化
    point_cloud_range = [-69.12, -39.68, -1, 69.12, 39.68, 7]
    
    # 获取点云文件列表
    lidar_files = sorted([f for f in os.listdir(args.lidar_dir) if f.endswith(args.ext)])
    
    # 处理每个点云文件
    for file_name in tqdm(lidar_files, desc="处理点云文件"):
        # 加载点云
        lidar_path = os.path.join(args.lidar_dir, file_name)
        points = load_lidar_file(lidar_path)
        
        try:
            # 准备数据
            points_tensor = torch.tensor(points, dtype=torch.float32).cuda()
            
            # 使用模型的simple_test方法
            with torch.no_grad():
                # 尝试不同的方法调用模型
                if hasattr(model, 'simple_test'):
                    outputs = model.simple_test([points_tensor], [{'token': file_name}])
                elif hasattr(model, 'forward'):
                    outputs = model.forward([points_tensor], return_loss=False)
                else:
                    print(f"模型没有simple_test或forward方法，无法处理文件 {file_name}")
                    continue
            
            # 检查outputs是否为None或空
            if outputs is None or len(outputs) == 0:
                print(f"处理文件 {file_name} 时模型没有输出结果")
                continue
                
            # 处理输出结果
            for output in outputs:
                for k, v in output.items():
                    if k not in ["metadata"] and isinstance(v, torch.Tensor):
                        output[k] = v.to(torch.device("cpu"))
            
            # 加载标签（如果有）
            gt_boxes = None
            if args.label_dir:
                label_file = os.path.join(args.label_dir, file_name.replace(args.ext, args.label_ext))
                gt_boxes = load_label_file(label_file)
            
            # 保存可视化结果
            base_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(output_dir, f"{base_name}.png")
            visualize_result(points, outputs[0], save_path, point_cloud_range, gt_boxes)
            
            # 可选：保存检测结果为文本文件
            result_path = os.path.join(output_dir, f"{base_name}_pred.txt")
            boxes = outputs[0]['box3d_lidar'].numpy()
            scores = outputs[0]['scores'].numpy()
            labels = outputs[0]['label_preds'].numpy()
            
            with open(result_path, 'w') as f:
                for i in range(len(boxes)):
                    if scores[i] > args.threshold:
                        box = boxes[i]
                        line = f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[8]} {scores[i]} {labels[i]}\n"
                        f.write(line)
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"推理完成，结果保存在 {output_dir}")

if __name__ == '__main__':
    main()