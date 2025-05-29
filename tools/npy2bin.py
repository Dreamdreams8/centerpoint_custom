import os
import numpy as np
from tqdm import tqdm

def convert_npy_to_bin(npyfolder, binfolder):
    """
    将NPY文件转换为BIN文件
    
    参数:
    npyfolder (str): 输入的NPY文件夹路径
    binfolder (str): 输出的BIN文件夹路径
    """
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, npyfolder)
    file_list = sorted(os.listdir(ori_path))  # 保持文件顺序一致
    des_path = os.path.join(current_path, binfolder)
    
    # 创建输出文件夹
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    
    # 遍历所有npy文件
    for file in tqdm(file_list):
        if file.endswith('.npy'):
            # 构造完整文件路径
            npy_file = os.path.join(ori_path, file)
            bin_file = os.path.join(des_path, file.replace('.npy', '.bin'))
            
            # 加载npy文件
            data = np.load(npy_file)
            # print("data shape: ",data.shape)
            # 保持原始维度，不进行维度转换
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # 直接保存为bin文件，保持原始形状
            data.tofile(bin_file)

if __name__ == '__main__':
    # 示例用法
    convert_npy_to_bin(
        "/media/why/4d64b9d8-e7cd-4079-bbb3-646afebfb8d4/centerpoint_custom/data/custom_0411_zzg/points_npy",
        "/media/why/4d64b9d8-e7cd-4079-bbb3-646afebfb8d4/centerpoint_custom/data/custom_0411_zzg/points"
    )
