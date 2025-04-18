import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ----------------------
# 可配置参数
# ----------------------
SELECTED_CATEGORIES = ['car', 'truck']  # 需要处理的类别
STEPS_PER_EPOCH = 3131                  # 每个epoch的步数
MOVING_AVG_WINDOW = 1000                 # 移动平均窗口大小
SHOW_RAW_DATA = True                    # 是否显示原始数据
LINE_COLORS = {
    'raw': '#808080',       # 原始数据颜色 (灰色)
    'moving_avg': '#FF4500' # 移动平均颜色 (橙红色)
}
OUTPUT_DIR = Path("loss_show")          # 输出目录

# ----------------------
# 初始化数据结构
# ----------------------
# element_names = ['x', 'y', 'z', 'l', 'w', 'h', 'vx', 'vy', 'theta_s', 'theta_c']
element_names = ['x', 'y', 'z', 'l', 'w', 'h', 'theta_s', 'theta_c']
loss_types = ['loss', 'hm_loss', 'loc_loss'] + element_names

data_structure = {
    'global_steps': [],
    'loss': [],
    'hm_loss': [],
    'loc_loss': [],
    'loc_loss_elements': {k: [] for k in element_names}
}
data = {cat: data_structure.copy() for cat in SELECTED_CATEGORIES}

# ----------------------
# 数据解析函数
# ----------------------
def parse_log(log_path):
    """解析日志文件并填充数据结构"""
    epoch_pattern = re.compile(r'Epoch \[(\d+)/(\d+)\]\[(\d+)/(\d+)\]')
    task_pattern = re.compile(
        r"task : \['(\w+)'\].*?loss: ([\d.]+)"
        r".*?hm_loss: ([\d.]+)"
        r".*?loc_loss: ([\d.]+)"
        r".*?loc_loss_elem: \[(.*?)\]"
    )

    current_epoch = 0
    with open(log_path, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                step_in_epoch = int(epoch_match.group(3))
                global_step = (current_epoch-1)*STEPS_PER_EPOCH + step_in_epoch
                continue
            
            task_match = task_pattern.search(line)
            if task_match:
                category = task_match.group(1).lower()
                if category not in SELECTED_CATEGORIES:
                    continue
                
                # 记录基础数据
                data[category]['global_steps'].append(global_step)
                data[category]['loss'].append(float(task_match.group(2)))
                data[category]['hm_loss'].append(float(task_match.group(3)))
                data[category]['loc_loss'].append(float(task_match.group(4)))
                
                # 解析定位元素
                elements = [float(x.strip("'")) for x in task_match.group(5).split(', ')]
                for idx, elem in enumerate(element_names):
                    data[category]['loc_loss_elements'][elem].append(elements[idx])

# ----------------------
# 可视化函数
# ----------------------
def calculate_moving_avg(values, window_size):
    """计算移动平均值"""
    cumsum = np.cumsum(np.insert(values, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def plot_and_save():
    """生成并保存所有损失曲线"""
    plt.style.use('seaborn')  # 使用更美观的样式
    
    for category in SELECTED_CATEGORIES:
        category_data = data[category]
        steps = np.array(category_data['global_steps'])
        
        for loss_type in loss_types:
            plt.figure(figsize=(12, 6))
            
            # 获取原始数据
            values = (
                category_data[loss_type] 
                if loss_type in ['loss', 'hm_loss', 'loc_loss'] 
                else category_data['loc_loss_elements'][loss_type]
            )
            values = np.array(values)
            
            # 绘制原始数据
            if SHOW_RAW_DATA and len(values) > 0:
                plt.plot(
                    steps, values,
                    color=LINE_COLORS['raw'],
                    alpha=0.3,
                    linewidth=1,
                    label='Raw Data'
                )
            
            # 计算并绘制移动平均
            if len(values) >= MOVING_AVG_WINDOW:
                moving_avg = calculate_moving_avg(values, MOVING_AVG_WINDOW)
                plt.plot(
                    steps[MOVING_AVG_WINDOW-1:],
                    moving_avg,
                    color=LINE_COLORS['moving_avg'],
                    linewidth=2,
                    label=f'{MOVING_AVG_WINDOW}-step Moving Avg'
                )
            
            # 图表装饰
            title_type = (
                loss_type if loss_type in element_names 
                else {'loss': 'Total', 'hm_loss': 'Heatmap', 'loc_loss': 'Localization'}[loss_type]
            )
            plt.title(f"{category.capitalize()} - {title_type} Loss")
            plt.xlabel("Global Training Step")
            plt.ylabel("Loss Value")
            plt.grid(True, alpha=0.2)
            plt.legend()
            
            # 保存图像
            output_path = OUTPUT_DIR / f"{category}_{loss_type}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 初始化环境
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 替换为实际日志路径
    parse_log("checkpoint/20250417_064359.log")
    plot_and_save()
    print(f"可视化结果已保存至：{OUTPUT_DIR.absolute()}")