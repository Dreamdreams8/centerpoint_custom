from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg') 
# dataroot为存放nuscenes的文件夹，4个文件夹放置一起
lidar_toptoken = '9d9bf11fb0e144c8b446d54a8a00184f'
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)
my_scene = nusc.scene[1]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
sensor = 'LIDAR_TOP'
lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor])
lidar_top_data = nusc.get('sample_data', lidar_toptoken)
nusc.render_sample_data(lidar_top_data['token'])

boxes = nusc.get_boxes(lidar_top_data['token'])
print("boxes: ",boxes)

# sensor = 'CAM_FRONT'
sensor = 'CAM_BACK'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
cam_front_data = nusc.get('sample_data', lidar_toptoken)
nusc.render_sample_data(cam_front_data['token'])
plt.show()  # 显式调用显示
lidar_path, boxes, _ = nusc.get_sample_data(my_sample['data']['LIDAR_TOP'])
# print("lidar_path: ",lidar_path)
# print("boxes: ",boxes)