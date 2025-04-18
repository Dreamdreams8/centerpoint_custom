import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
import argparse
from det3d.core import box_np_ops

# 定义类别映射，将自定义类别映射到NuScenes类别
CLASS_MAP = {
    "Car": "car",
    "Truck": "truck",
    # 可以根据需要添加更多映射
}

def create_custom_infos(root_path, split='train', nsweeps=1, filter_zero=True):
    """
    创建与NuScenes格式兼容的自定义数据集info文件
    
    Args:
        root_path: 数据集根目录
        split: 'train'或'val'
        nsweeps: 点云帧数（对于自定义数据集，我们只有单帧，所以设为1）
        filter_zero: 是否过滤零点的标注
    """
    root_path = Path(root_path)
    custom_path = root_path
    
    # 读取分割文件
    split_file = custom_path / 'ImageSets' / f'{split}.txt'
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {split_file} not found")
    
    with open(split_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(sample_ids)} samples for {split}")
    
    # 创建info列表
    custom_infos = []
    
    # 处理每个样本
    for sample_id in tqdm(sample_ids):
        # 检查点云文件是否存在
        lidar_path = custom_path / 'points' / f'{sample_id}.bin'
        if not lidar_path.exists():
            print(f"Warning: Point cloud file {lidar_path} not found for {sample_id}")
            continue
        
        # 检查标签文件是否存在
        label_path = custom_path / 'labels' / f'{sample_id}.txt'
        if not label_path.exists():
            print(f"Warning: Label file {label_path} not found for {sample_id}")
            continue
        
        # 直接使用原始点云文件路径
        lidar_path_rel = 'points/' + f'{sample_id}.bin'
        
        # 读取标签
        boxes = []
        names = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                
                # 解析标签
                x, y, z, l, w, h, yaw = map(float, parts[:7])
                category = parts[7].lower()  # 转换为小写
                
                # 创建边界框 [x, y, z, w, l, h, yaw, vx, vy]
                # 注意：CenterPoint使用的边界框格式是[x, y, z, w, l, h, yaw, vx, vy]
                box = [x, y, z, l, w, h, 0, 0, yaw]  # 速度设为0
                # box = [x, y, z, l, w, h, yaw]  # 速度设为0    
                boxes.append(box)
                names.append(category)
        
        if not boxes:
            print(f"Warning: No valid boxes found for {sample_id}")
            continue
        
        # 转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32)
        names = np.array(names)
        
        # 创建虚拟的相机路径和内参
        cam_front_path = str(custom_path / 'virtual_cam' / f'{sample_id}.jpg')
        cam_intrinsic = np.array([
            [800.0, 0.0, 800.0],
            [0.0, 800.0, 450.0],
            [0.0, 0.0, 1.0]
        ])
        
        # 创建虚拟的坐标变换矩阵（单位矩阵，表示坐标系相同）
        ref_from_car = np.eye(4)
        car_from_global = np.eye(4)
        
        # 创建虚拟的相机到激光雷达的变换
        all_cams_from_lidar = {
            'CAM_FRONT': np.eye(4),
            'CAM_FRONT_RIGHT': np.eye(4),
            'CAM_FRONT_LEFT': np.eye(4),
            'CAM_BACK': np.eye(4),
            'CAM_BACK_LEFT': np.eye(4),
            'CAM_BACK_RIGHT': np.eye(4)
        }
        
        # 创建虚拟的相机内参
        all_cams_intrinsic = {cam: cam_intrinsic for cam in all_cams_from_lidar.keys()}
        
        # 创建虚拟的相机路径
        all_cams_path = {cam: cam_front_path for cam in all_cams_from_lidar.keys()}
        
        # 在创建虚拟相机路径之前添加
        virtual_cam_dir = custom_path / 'virtual_cam'
        virtual_cam_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建info字典
        info = {
            "token": sample_id,
            "lidar_path": lidar_path,  # 使用相对路径
            "cam_front_path": 'virtual_cam/virtual.jpg',
            "cam_intrinsic": cam_intrinsic,
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": 0,
            "all_cams_from_lidar": all_cams_from_lidar,
            "all_cams_intrinsic": all_cams_intrinsic,
            "all_cams_path": all_cams_path,
            "gt_boxes": boxes,
            "gt_names": names,
            "gt_boxes_token": np.array([f"{sample_id}_{i}" for i in range(len(boxes))]),
            "gt_boxes_velocity": np.zeros((len(boxes), 3), dtype=np.float32)  # 速度设为0
        }
        
        custom_infos.append(info)
    
    # 保存info文件
    print(f"Saving {len(custom_infos)} {split} infos to {root_path}")
    with open(root_path / f"infos_{split}_{nsweeps:02d}sweeps_withvelo_filter_{filter_zero}.pkl", "wb") as f:
        pickle.dump(custom_infos, f)
    
    return custom_infos

def create_custom_groundtruth_database(data_path, info_path, used_classes=None, database_save_path=None, db_info_save_path=None):
    """创建自定义数据集的GT数据库"""
    print(f"Create GT Database using {info_path}")
    print(f"Used classes: {used_classes}")
    
    database_save_path = Path(database_save_path)
    db_info_save_path = Path(db_info_save_path)
    database_save_path.mkdir(parents=True, exist_ok=True)
    
    all_db_infos = {}
    for name in used_classes:
        all_db_infos[name] = []
    
    # 加载info文件
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    # 处理每个样本
    for k in tqdm(range(len(infos))):
        info = infos[k]
        sample_idx = info["token"]
        
        # 读取点云
        lidar_path = info["lidar_path"]
        if not lidar_path.exists():
            print(f"Warning: Point cloud file {lidar_path} not found")
            continue
        
        try:
            points = np.fromfile(str(lidar_path), dtype=np.float32)
            points = points.reshape(-1, 4)
                # 添加第五维
            ring_index = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, ring_index])            
            # 确定点云维度
            # if points.shape[0] % 5 == 0:
            #     points = points.reshape(-1, 5)
            # elif points.shape[0] % 4 == 0:
            #     points = points.reshape(-1, 4)
            #     # 添加第五维
            #     ring_index = np.zeros((points.shape[0], 1), dtype=np.float32)
            #     points = np.hstack([points, ring_index])
            # else:
            #     print(f"Warning: Cannot determine point cloud dimension for {lidar_path}")
            #     continue
        except Exception as e:
            print(f"Error reading point cloud {lidar_path}: {e}")
            continue
        
        # 获取标注
        annotations = info["gt_boxes"]
        gt_names = info["gt_names"]
        
        # 确保标注不为空
        if len(annotations) == 0:
            continue
        
        # 处理每个目标
        for i, name in enumerate(gt_names):
            # 类别名称已经是小写了，不需要再转换
            if name not in used_classes:
                continue
            
            # 获取边界框
            box = annotations[i]
            
            # 计算点是否在边界框内
            try:
                # 使用自定义函数计算点是否在边界框内
                mask = points_in_box(points, box)
                
                # 获取边界框内的点
                gt_points = points[mask]
                
                # 如果边界框内没有点，跳过
                if gt_points.shape[0] == 0:
                    continue
                
                # 创建保存路径
                filename = f"{sample_idx}_{name}_{i}.bin"
                filepath = database_save_path / name / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存点云
                with open(filepath, 'wb') as f:
                    gt_points.astype(np.float32).tofile(f)
                
                # 创建数据库信息
                db_info = {
                    "name": name,
                    "path": str(filepath.relative_to(data_path)),
                    "image_idx": sample_idx,
                    "gt_idx": i,
                    "box3d_lidar": box,
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": 0,
                    "group_id": i,
                }
                
                # 添加到数据库信息字典
                all_db_infos[name].append(db_info)
            except Exception as e:
                print(f"Error processing box {i} for {sample_idx}: {e}")
                continue
    
    # 输出每个类别的数量
    for k, v in all_db_infos.items():
        print(f"Database {k}: {len(v)}")
    
    # 保存数据库信息
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
    
    print(f"GT Database saved to {database_save_path}")
    print(f"GT Database info saved to {db_info_save_path}")

def custom_data_prep(root_path, nsweeps=1, filter_zero=True):
    """
    准备自定义数据集
    
    Args:
        root_path: 数据集根目录
        nsweeps: 点云帧数
        filter_zero: 是否过滤零点的标注
    """
    root_path = Path(root_path)
    
    # 创建训练集info文件
    train_infos = create_custom_infos(
        root_path=root_path,
        split="train",
        nsweeps=nsweeps,
        filter_zero=filter_zero
    )
    
    # 创建验证集info文件
    val_infos = create_custom_infos(
        root_path=root_path,
        split="val",
        nsweeps=nsweeps,
        filter_zero=filter_zero
    )
    
    # 创建ground truth数据库 - 使用小写类别名称
    create_custom_groundtruth_database(
        data_path=root_path,
        info_path=root_path / f"infos_train_{nsweeps:02d}sweeps_withvelo_filter_{filter_zero}.pkl",
        used_classes=["car", "truck"],  # 使用小写类别名称
        database_save_path=root_path / f"gt_database_{nsweeps:02d}sweeps_withvelo",
        db_info_save_path=root_path / f"dbinfos_train_{nsweeps:02d}sweeps_withvelo.pkl"
    )

def points_in_box(points, box):
    """简单计算点是否在边界框内"""
    # 提取边界框参数
    x, y, z, w, l, h, yaw = box[:7]
    
    # 计算边界框的半宽、半长、半高
    half_w = w / 2
    half_l = l / 2
    half_h = h / 2
    
    # 计算点到边界框中心的距离
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    dz = points[:, 2] - z
    
    # 旋转点到边界框坐标系
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    dx_rot = dx * cos_yaw + dy * sin_yaw
    dy_rot = -dx * sin_yaw + dy * cos_yaw
    
    # 判断点是否在边界框内
    mask_x = np.abs(dx_rot) <= half_l
    mask_y = np.abs(dy_rot) <= half_w
    mask_z = np.abs(dz) <= half_h
    
    mask = mask_x & mask_y & mask_z
    
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom dataset for CenterPoint")
    parser.add_argument("--root_path", type=str, required=True, help="Path to custom dataset")
    parser.add_argument("--nsweeps", type=int, default=1, help="Number of sweeps")
    parser.add_argument("--filter_zero", action="store_true", help="Filter zero annotations")
    
    args = parser.parse_args()
    custom_data_prep(args.root_path, args.nsweeps, args.filter_zero)