import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

# 根据您的数据集调整类别
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["truck"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,  # 使用5维点云
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8  # 使用5维点云
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        # common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)},
        share_conv_channel=64,
        dcn_head=False
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)

train_cfg = dict(
    assigner=assigner,
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075]
)

# dataset settings
dataset_type = "NuScenesDataset"  # 使用NuScenes数据集格式
nsweeps = 1  # 每个样本使用的雷达扫描次数
data_root = "data/custom_0411_zzg"  # 数据集根目录
# data_root = "data/custom" 

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,  # 随机打乱点云
    global_rot_noise=[-0.785, 0.785],    # 随机旋转范围，约±45度
    global_scale_noise=[0.90, 1.10],     # 随机缩放范围，85%~115%
    global_translate_std=0.5,            # 随机平移标准差
    db_sampler=dict(
        type="GT-AUG",                   # 使用GT数据库增强
        enable=False,
        db_info_path=data_root + "/dbinfos_train_01sweeps_withvelo.pkl",
        sample_groups=[
            dict(car=2),    # 每个场景额外采样2个汽车
            dict(truck=3),  # 每个场景额外采样3个卡车
        ],
        db_prep_steps=[
            dict(filter_by_min_num_points=dict(car=5, truck=5)),  # 过滤掉点数过少的样本
        ],
        global_random_rotation_range_per_object=[0, 0],
        rate=1.0,
    ),
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    class_names=class_names,
)

voxel_generator = dict(
    range=[-54, -54, -1.0, 54, 54, 7.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=assigner),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=assigner),
    dict(type="Reformat"),
]

train_anno = data_root + "/infos_train_01sweeps_withvelo_filter_True.pkl"
val_anno = data_root + "/infos_val_01sweeps_withvelo_filter_True.pkl"
test_anno = data_root + "/infos_val_01sweeps_withvelo_filter_True.pkl"

data = dict(
    samples_per_gpu=10,     # 每个GPU的批量大小
    workers_per_gpu=8,      # 每个GPU的数据加载线程数
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno, 
        test_mode=True,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam",           # 使用Adam优化器
    amsgrad=0.0,          # 不使用AMSGrad变体
    wd=0.01,             # 权重衰减系数，用于防止过拟合
    fixed_wd=True,       # 固定权重衰减
    moving_average=False, # 不使用滑动平均
)
lr_config = dict(
    type="one_cycle",      # 使用one cycle学习率策略
    lr_max=0.001,         # 最大学习率
    moms=[0.95, 0.85],    # momentum的范围
    div_factor=10,        # 初始学习率 = lr_max/div_factor = 0.0003
    pct_start=0.4,        # 预热阶段占总轮数的40%，即20轮
)

# 检查点配置
checkpoint_config = dict(
    interval=1,           # 每1轮保存一次模型
)
# yapf:disable
log_config = dict(
    interval=5,           # 每5次迭代打印一次日志
    hooks=[
        dict(type="TextLoggerHook"),  # 文本日志
        # dict(type="TensorboardLoggerHook")  # Tensorboard日志(已注释)
    ],
)
# yapf:enable
# runtime settings
total_epochs = 80         # 总训练轮数
device_ids = range(8)     # 使用的GPU设备ID
dist_params = dict(backend="nccl", init_method="env://")  # 分布式训练参数
log_level = "INFO"        # 日志级别
work_dir = './work_dirs/custom_centerpoint_voxelnet_0418'  # 工作目录

# 从中断处继续训练
# load_from = 'work_dirs/custom_centerpoint_voxelnet_0416/epoch_20.pth'  # 加载预训练模型
load_from = None
resume_from = None  # 不从中断处继续训练
workflow = [('train', 1)]  # 训练流程

warmup_config = dict(
    type='linear',         # 线性预热
    warmup_iters=80,      # 预热迭代次数
    warmup_ratio=1.0 / 3,  # 预热初始学习率比例
)