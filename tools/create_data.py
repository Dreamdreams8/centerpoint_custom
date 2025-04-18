import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds
from tools.custom_data_prep import custom_data_prep  # 导入自定义数据集处理模块

# root_path = data/custom
def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, virtual=False):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    if version == 'v1.0-mini':
        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
            nsweeps=nsweeps,
            virtual=virtual
        )

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )

# 添加自定义数据集处理函数
def custom_dataset_prep(root_path, nsweeps=1, filter_zero=True):
    custom_data_prep(root_path, nsweeps=nsweeps, filter_zero=filter_zero)

if __name__ == "__main__":
    fire.Fire()

# python tools/create_data.py custom_dataset_prep --root_path=data/custom_0411_zzg --nsweeps=1 --filter_zero=True
# python tools/create_data.py nuscenes_data_prep --root_path=data/nuscenes --version=v1.0-mini  --nsweeps=1 --filter_zero=True