from det3d.datasets.nuscenes.nuscenes import NuScenesDataset
from det3d.datasets.waymo.waymo import WaymoDataset

dataset_factory = {
    # "NuScenesDataset": NuScenesDataset,
    # "WaymoDataset": WaymoDataset,
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
