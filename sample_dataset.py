import os
import scipy.io as scio
from py_utils import sampling, generate_cube
from py_utils import mapset_normalization
from config import dataset_config
import numpy as np

# TODO: here to change dataset
dataset_name = 'muufl'
# dataset_name = 'augsburg'

if dataset_name in dataset_config:
    config = dataset_config[dataset_name]
    dataset_dir = config['dataset_dir']
    concate_pixel = config['concate_pixel']
    class_labels = config['class_labels']
else:
    raise ValueError("Unknown dataset")

num_class = max(class_labels.keys())

try:
    HSI_MapSet = scio.loadmat(os.path.join(dataset_dir, f'HSI.mat'))['HSI']
    LiDAR_MapSet = scio.loadmat(os.path.join(dataset_dir, f'LiDAR.mat'))['LiDAR']
except:
    HSI_MapSet = scio.loadmat(os.path.join(dataset_dir, f'data_HS_LR.mat'))['data_HS_LR']
    LiDAR_MapSet = scio.loadmat(os.path.join(dataset_dir, f'data_DSM.mat'))['data_DSM']

try:
    TrLabel = scio.loadmat(os.path.join(dataset_dir, f'TrLabel.mat'))['TrLabel'].astype(np.int8) 
    TeLabel = scio.loadmat(os.path.join(dataset_dir, f'TeLabel.mat'))['TeLabel'].astype(np.int8) 
except:
    TrLabel = scio.loadmat(os.path.join(dataset_dir, f'TrLabel.mat'))['TRLabel'].astype(np.int8) 
    TeLabel = scio.loadmat(os.path.join(dataset_dir, f'TeLabel.mat'))['TSLabel'].astype(np.int8) 

os.makedirs(os.path.join(dataset_dir, f'TrTe'), exist_ok = True)
f = open(os.path.join(dataset_dir, f'TrTe/info.txt'), 'w')

print('HSI_MapSet.shape = ', HSI_MapSet.shape, file = f)
print('LiDAR_MapSet.shape = ',LiDAR_MapSet.shape, file = f)
print('TrLabel.shape = ',TrLabel.shape, file = f)
print('TeLabel.shape = ',TeLabel.shape, file = f)
print("Unique values in TrLabel:", np.unique(TrLabel))
print("Unique values in TeLabel:", np.unique(TeLabel))

HSI_MapSet, row, col, _ = mapset_normalization(mapset = HSI_MapSet, concate_pixel = concate_pixel, type = 'HSI')
LiDAR_MapSet, row, col, _ = mapset_normalization(mapset = LiDAR_MapSet, concate_pixel = concate_pixel, type = 'LiDAR', n_feature = 1)

TrLabel = TrLabel.reshape(row * col)
TeLabel = TeLabel.reshape(row * col)

Trlabel_flat = TrLabel.flatten()
Telabel_flat = TeLabel.flatten()


undefined_value = min(np.unique(Trlabel_flat))

# 统计每个类别的数量,忽略未标注的像素（-1）
Tr_counts = np.bincount(Trlabel_flat[Trlabel_flat != undefined_value])
Te_counts = np.bincount(Telabel_flat[Telabel_flat != undefined_value])

# 打印统计结果
print("TrLabel 中每个类别的数量:", file = f)
for i in range(1, len(Tr_counts)):
    print(f"类别 {i}: {Tr_counts[i]} 个", file = f)

print("\nTelabel 中每个类别的数量:", file = f)
for i in range(1, len(Te_counts)):
    print(f"类别 {i}: {Te_counts[i]} 个", file = f)

if dataset_name == 'houston13':
    f.close()
    print('houston13 can not be sampled, because we only have LiDAR.map with 1 band')
    exit()

patch_size = 7
train_idx, test_idx = sampling(TrLabel, TeLabel)
# 打印生成的索引
print("训练集索引数量:", len(train_idx), file = f)
print("测试集索引数量:", len(test_idx), file = f)

HSI_TrSet, Y_train = generate_cube(train_idx, HSI_MapSet, TrLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)
HSI_TeSet, Y_test = generate_cube(test_idx, HSI_MapSet, TeLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)
LiDAR_TrSet, _ = generate_cube(train_idx, LiDAR_MapSet, TrLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)
LiDAR_TeSet, _ = generate_cube(test_idx, LiDAR_MapSet, TeLabel.reshape(row * col), patch_size, row, col, num_class = num_class, shuffle=False)


scio.savemat(os.path.join(dataset_dir, f'TrTe/HSI_TrSet.mat'), {'Data': HSI_TrSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/LiDAR_TrSet.mat'), {'Data': LiDAR_TrSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/Y_train.mat'), {'Data': Y_train})
scio.savemat(os.path.join(dataset_dir, f'TrTe/HSI_TeSet.mat'), {'Data': HSI_TeSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/LiDAR_TeSet.mat'), {'Data': LiDAR_TeSet.astype(np.float32)})
scio.savemat(os.path.join(dataset_dir, f'TrTe/Y_test.mat'), {'Data': Y_test})


print('HSI_TrSet.shape = ', HSI_TrSet.shape, file = f)
print('HSI_TeSet.shape = ', HSI_TeSet.shape, file = f)
print('LiDAR_TrSet.shape = ', LiDAR_TrSet.shape, file = f)
print('LiDAR_TeSet.shape = ', LiDAR_TeSet.shape, file = f)
print('Y_train.shape = ', Y_train.shape, file = f)
print('Y_test.shape = ', Y_test.shape, file = f)
f.close()