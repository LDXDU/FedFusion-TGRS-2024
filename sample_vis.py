import numpy as np
import os
import scipy.io
from config import dataset_config
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns


# TODO: here to change dataset
# dataset_name = 'trento'
# dataset_name = 'houston13'
# dataset_name = 'muufl'
dataset_name = 'augsburg'

if dataset_name in dataset_config:
    config = dataset_config[dataset_name]
    dataset_dir = config['dataset_dir']
    class_labels = config['class_labels']
else:
    raise ValueError("Unknown dataset")



cm_labels = list(class_labels.values())[1:]  # 不包括Undefined
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", len(cm_labels))):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
colormap = [np.array(color) / 255.0 for color in palette.values()]

ground_truth_labels = scipy.io.loadmat(os.path.join(dataset_dir, f'gt.mat'))['gt'].astype(np.int8) 
TrLabel = scipy.io.loadmat(os.path.join(dataset_dir, f'TrLabel.mat'))
TeLabel = scipy.io.loadmat(os.path.join(dataset_dir, f'TeLabel.mat'))

try:
    TrLabel = TrLabel['TrLabel']
    TeLabel = TeLabel['TeLabel']
except:
    TrLabel = TrLabel['TRLabel']
    TeLabel = TeLabel['TSLabel']


cmap = ListedColormap(colormap)

ticks = [0 if key == -1 else key for key in sorted(class_labels.keys())]  # 获取所有类别的键, 并将 -1 映射为 0
ticklabels = [class_labels[key] for key in sorted(class_labels.keys())] # 设置颜色条的刻度标签
print(ticklabels)

# 可视化 Ground Truth 标签
plt.figure(figsize=(10, 8))
plt.imshow(ground_truth_labels, cmap=cmap, interpolation='none', vmin=0)
cbar = plt.colorbar(ticks=ticks, label='Class Labels', boundaries=np.arange(-0.5, len(ticks) + 0.5))
cbar.set_ticklabels(ticklabels)
plt.title('Ground Truth Labels (' + dataset_name.upper() + ' Dataset)')
plt.savefig(os.path.join(dataset_dir, f'ground_truth_labels.png'), bbox_inches='tight', dpi=300)  # 保存图像

plt.figure(figsize=(10, 8))
plt.imshow(TeLabel, cmap=cmap, interpolation='none', vmin=0)
cbar = plt.colorbar(ticks=ticks, label='Class Labels', boundaries=np.arange(-0.5, len(ticks) + 0.5))
cbar.set_ticklabels(ticklabels)
plt.title('Test Set Labels (' + dataset_name.upper() + ' Dataset)')
plt.savefig(os.path.join(dataset_dir, f'test_set_labels.png'), bbox_inches='tight', dpi=300)  # 保存图像

plt.figure(figsize=(10, 8))
plt.imshow(TrLabel, cmap=cmap, interpolation='none', vmin=0)
cbar = plt.colorbar(ticks=ticks, label='Class Labels', boundaries=np.arange(-0.5, len(ticks) + 0.5))
cbar.set_ticklabels(ticklabels)
plt.title('Train Set Labels (' + dataset_name.upper() + ' Dataset)')
plt.savefig(os.path.join(dataset_dir, f'train_set_labels.png'), bbox_inches='tight', dpi=300)  # 保存图像