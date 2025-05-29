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
from shapely.geometry import Polygon

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

# 参数配置
IOU_THRESHOLD = 0.5  # IoU阈值
USE_BEV = True       # 使用鸟瞰图IoU计算    

key_class = ['Car', 'Truck']  # 要评估的类别
class_mapping = {
    0: 'Car',
    1: 'Truck',
    2: 'Lockbox',
    3: 'Pedestrian',
    4: 'Lockstation',
    5: 'Bridge',
    6: 'ForkLift',
    7: 'Conetank'
}
class2int_mapping = {
    'Car': 0,
    'Truck': 1,
    'Lockbox': 2,
    'Pedestrian': 3,
    'Lockstation': 4,
    'Bridge': 5,
    'ForkLift': 6,
    'Conetank': 7,
    'IGV':8
}
# 从result_show_2d.py引入的可视化相关代码
box_colormap = [
    [1, 0, 0],    # Car (红色)
    [0, 1, 0],    # Truck (绿色)
    [0, 0, 1],    # Lockbox (蓝色)
    [1, 1, 0],  # Pedestrian (黄色)
    [1, 0, 1],  # Lockstation (品红)
    [0, 1, 1],  # Bridge (青色)
    [0.5, 0, 0],    # ForkLift (深红色)
    [0, 0.5, 0],    # Conetank (深绿色)
    [0, 0, 0.5]     # IGV (深蓝色)
]

def draw_2d_box(ax, x, y, w, h, theta_rad, color='r',linestyles='-'):
    """绘制2D边界框"""
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad), np.cos(theta_rad)]])
    
    corners = np.array([[-w/2, -h/2],
                       [w/2, -h/2],
                       [w/2, h/2],
                       [-w/2, h/2]])
    
    rotated_corners = np.dot(corners, R.T)
    rotated_corners += np.array([x, y])
    
    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=2, linestyle=linestyles)
    ax.plot([rotated_corners[0, 0], rotated_corners[-1, 0]],
            [rotated_corners[0, 1], rotated_corners[-1, 1]], color=color, linewidth=2, linestyle=linestyles)
    # Add heading direction indicator (front of the vehicle)
    front_point = np.array([[w/2, 0]])  # Front center point in local coordinates
    rotated_front = np.dot(front_point, R.T) + np.array([x, y])
    ax.plot([x, rotated_front[0, 0]], 
            [y, rotated_front[0, 1]], 
            color=color, linewidth=2, linestyle='--')    

def visualize_gt_pred_result(points, detection, gt_boxes,save_path, point_cloud_range):
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
    for box in gt_boxes:
        color = box_colormap[class2int_mapping[box['class']]]
        draw_2d_box(ax, box['x'], box['y'],
                   box['w'], box['l']+1.0, box['angle'], color,linestyles=':')  # 为了区分将真值框的宽度增加1.0m        
    # 保存图片
    plt.axis('off')
    plt.savefig(save_path, dpi=50, bbox_inches='tight', pad_inches=0, 
                facecolor='black', format='png')
    plt.close()


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

def load_label(file_path):
    results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        parts = line.strip().split()
        # 确保每行有9个元素，因为你的字典要求有9个键值对
        if len(parts) != 8:
            print(f"Warning: line skipped due to incorrect format - {line}")
            continue
            
        result = {
            'x': float(parts[0]),
            'y': float(parts[1]),
            'z': float(parts[2]),
            'w': float(parts[3]),
            'l': float(parts[4]),
            'h': float(parts[5]),
            'angle': float(parts[6]),  # 弧度
            # 'class': parts[7].lower()  # 假设你想要保留类别为字符串类型
            'class': parts[7]
            # 'score': float(parts[8])
        }
        results.append(result)
    
    return results

def compute_bev_iou(box1, box2):
    """计算鸟瞰图下的IoU"""
    def get_vertices(x, y, w, l, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        half_w = w / 2
        half_l = l / 2
        corners = np.array([
            [-half_w, -half_l],
            [half_w, -half_l],
            [half_w, half_l],
            [-half_w, half_l]
        ])
        rotated = np.dot(corners, np.array([[cos_a, -sin_a], [sin_a, cos_a]]))
        rotated[:, 0] += x
        rotated[:, 1] += y
        return rotated
    
    try:
        poly1 = Polygon(get_vertices(box1['x'], box1['y'], box1['w'], box1['l'], box1['angle']))
        poly2 = Polygon(get_vertices(box2['x'], box2['y'], box2['w'], box2['l'], box2['angle']))
    except:
        return 0.0

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union != 0 else 0.0

def compute_ap(recall, precision):
    """计算AP PASCAL VOC 11点法 """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

def get_map(class_stats,all_preds):
    # TP FP FN
    total_tp = sum(s['tp'] for s in class_stats.values())
    total_fp = sum(s['fp'] for s in class_stats.values())
    total_fn = sum(s['fn'] for s in class_stats.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算AP和mAP
    aps = []
    for c in class_stats:
        # 过滤无效类别
        if class_stats[c]['total_gt'] == 0:
            continue
        
        # 获取当前类别的所有预测
        class_preds = [p for p in all_preds if p['class'] == c]
        class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
        
        # 提取TP/FP标记
        tp_list = np.array([p['is_tp'] for p in class_preds])
        fp_list = np.logical_not(tp_list)
        
        # 计算累积TP/FP
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        
        # 计算precision和recall
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        recalls = tp_cum / class_stats[c]['total_gt']
        
        # 计算AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)

        # print(f'Class {class_mapping.get(c):<8} AP: {ap:.4f}')

    mAP = np.mean(aps) if aps else 0
    return precision, recall, f1, mAP    

def calculate_all_preds(gt_boxes, pred_boxes,class_stats,all_preds):
    # 按类别组织数据
    gt_dict = {}
    for gt in gt_boxes:
        c = gt['class']
        if c in key_class:
            gt_dict.setdefault(c, []).append(gt)
    
    pred_dict = {}
    for pred in pred_boxes:
        c = pred['class']
        if float(pred['score']) < 0.5:
            continue        
        if c in key_class:
            pred_dict.setdefault(c, []).append(pred)

    # 处理每个类别
    for c in set(gt_dict.keys()).union(pred_dict.keys()):
        # 跳过无效类别
        if c not in key_class:
            continue
        # 初始化类别统计
        if c not in class_stats:
            class_stats[c] = {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0}
        
        # 获取当前类别的预测和真值
        class_preds = sorted(pred_dict.get(c, []), key=lambda x: x['score'], reverse=True)
        class_gts = gt_dict.get(c, [])
        
        # 更新总真值数
        class_stats[c]['total_gt'] += len(class_gts)
        
        # 初始化匹配状态
        matched = [False] * len(class_gts)
        
        # 处理每个预测
        for pred in class_preds:
            best_iou = 0.0
            best_idx = -1
            
            # 寻找最佳匹配
            for i, gt in enumerate(class_gts):
                if not matched[i]:
                    iou = compute_bev_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
            
            # 判断TP/FP
            if best_iou >= IOU_THRESHOLD and best_idx != -1:
                matched[best_idx] = True
                all_preds.append({'class': c, 'score': pred['score'], 'is_tp': True})
                class_stats[c]['tp'] += 1
            else:
                all_preds.append({'class': c, 'score': pred['score'], 'is_tp': False})
                class_stats[c]['fp'] += 1
        
        # 统计FN
        class_stats[c]['fn'] += len(class_gts) - sum(matched)






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
    checkpoint = load_checkpoint(model, 'work_dirs/custom_centerpoint_voxelnet_0421/epoch_269.pth', map_location="cpu")
    model.eval()
    model = model.cuda()
    
    cpu_device = torch.device("cpu")
    
    # 创建输出目录
    out_dir = "data/custom_0411_zzg/visualization_0422_269_epoch"
    pred_dir = os.path.join(out_dir, "pred")  # 预测结果目录
    gt_dir = os.path.join(out_dir, "gt")      # 标签结果目录
    
    for d in [pred_dir, gt_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # 获取点云范围用于可视化
    point_cloud_range = [-69.12, -39.68, -1, 69.12, 39.68, 7]
    
    # 处理每个样本
    class_stats = {}
    all_preds = []        
    for j, data_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        # if j >= 100:
        #     break
        # 获取点云数据
        points = data_batch['points'][0][:, 0:3].cpu().numpy()
        
        # ['metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates']
        # 获取点云文件名
        lidar_token = data_batch['metadata'][0]['token']
        # print(data_batch['metadata'])
        # print("lidar_token: ", str(data_batch['metadata'][0]['image_prefix']) + "/" +"labels/" + lidar_token + ".txt")
        label_path = str(data_batch['metadata'][0]['image_prefix']) + "/" +"labels/" + lidar_token + ".txt"
        gt_boxes = load_label(label_path)

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
        boxes = outputs[0]['box3d_lidar'].numpy()
        scores = outputs[0]['scores'].numpy()
        labels = outputs[0]['label_preds'].numpy()
        pred_boxes = [] 
        for i in range(len(boxes)):
            # print("type:",type(labels[i]))
            # print("str(labels[i]):  ",class_mapping[labels[i]])
            pred_boxes.append({
                'x': boxes[i, 0],  # x坐标
                'y': boxes[i, 1],  # y坐标
                'z': boxes[i, 2],  # z坐标
                'w': boxes[i, 3],  # 宽度
                'l': boxes[i, 4],  # 长度
                'h': boxes[i, 5],  # 高度
                'angle': boxes[i, 6],  # 旋转角度(弧度)
                'class': class_mapping[labels[i]],  # 类别ID转为字符串
                'score': scores[i]  # 置信度分数
            })        
        # print("pred_boxes: ",pred_boxes)
        # print("gt_boxes: ",gt_boxes)    
        calculate_all_preds(gt_boxes, pred_boxes,class_stats,all_preds)
        # # 保存预测结果可视化
        pred_save_path = os.path.join(pred_dir, f"{lidar_token}.png")
        visualize_gt_pred_result(points.T, outputs[0], gt_boxes,pred_save_path, point_cloud_range)  
    # print("class_stats: ",class_stats)      
    # print("all_preds: ",all_preds)
    precision, recall, f1, mAP = get_map(class_stats,all_preds)
    print(f'{"指标":<20} {"结果":>6}')
    print('-' * 32)
    print(f'{"Precision":<20} {precision:>10.4f}')
    print(f'{"Recall":<20} {recall:>10.4f}')
    print(f'{"F1 Score":<20} {f1:>10.4f}')
    print(f'{"mAP":<20} {mAP:>10.4f}')        

        

if __name__ == '__main__':
    main()

# python tools/demo_result_show_2d.py 