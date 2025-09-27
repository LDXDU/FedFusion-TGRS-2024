import numpy as np
import torch.nn.functional as F
import scipy.io as scio
import torch
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LambdaLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from py_utils import same_seeds, generate_batch, random_mini_batches_standardtwoModality
from py_utils import create_excel, log_and_print, mapset_normalization, result_cal, save_result_excel
import seaborn as sns
import argparse
import logging
import os
import datetime
from config import dataset_config, custom_colors
from model import Fed_Fusion, Fed_Fusion_Loss
import torch.distributed as dist
from matplotlib.colors import ListedColormap

# Argument parser for command-line options
def command_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', type=str, default='Fed_Fusion', help="Choose the model to use.")
    parser.add_argument('--mode', type=str, default='MML', choices=['MML', 'CML-LiDAR', 'CML-HSI'], help="Choose the model to use.")
    parser.add_argument('--inference_only', type=int, default='0')
    parser.add_argument('--with_svd', type=int, default='0', help="Use svd or not.")
    parser.add_argument('--svd_k', type=int, default='4', help="Only valid when with_svd is True.")
    parser.add_argument('--dataset_name', type=str, default='houston13', help="Dataset name.")
    parser.add_argument('--device', type=str, default='cuda', help="cpu or cuda")
    parser.add_argument('--output_dir', type=str, default='./output/debug', help="Directory to save outputs.")
    parser.add_argument('--save_excel', type=int, default=1, help="Set to True to save results in an Excel file.")
    parser.add_argument('--save_png', type=int, default=1, help="Set to True to save PNG images.")
    parser.add_argument('--log_path', type=str, default='./logs/debug.log', help="Path to save the log file.")
    parser.add_argument('--log_epoch', type=int, default=50, help="epochs interval to log.")
    args = parser.parse_args()

    return args

def load_dataset(batch_size):
    HSI_TrSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/HSI_TrSet.mat'))['Data'] # 加载数据集处理好的数据集
    HSI_TeSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/HSI_TeSet.mat'))['Data']
    LiDAR_TrSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/LiDAR_TrSet.mat'))['Data']
    LiDAR_TeSet = scio.loadmat(os.path.join(dataset_dir, f'TrTe/LiDAR_TeSet.mat'))['Data']
    Y_train = scio.loadmat(os.path.join(dataset_dir, f'TrTe/Y_train.mat'))['Data']
    Y_test = scio.loadmat(os.path.join(dataset_dir, f'TrTe/Y_test.mat'))['Data']
    log_and_print('HSI_TrSet.shape = ', HSI_TrSet.shape)
    log_and_print('HSI_TeSet.shape = ', HSI_TeSet.shape)
    log_and_print('LiDAR_TrSet.shape = ', LiDAR_TrSet.shape)
    log_and_print('LiDAR_TeSet.shape = ', LiDAR_TeSet.shape)
    log_and_print('Y_train.shape = ', Y_train.shape)
    log_and_print('Y_test.shape = ', Y_test.shape)

    trainset1 = HSI_TrSet
    trainset2 = LiDAR_TrSet
    if mode == 'MML':
        valset1 = HSI_TeSet
        valset2 = LiDAR_TeSet

    if mode ==  'CML-LiDAR':
        valset1 = np.zeros_like(HSI_TeSet)
        valset2 = LiDAR_TeSet

    if mode == 'CML-HSI':
        valset1 = HSI_TeSet
        valset2 = np.zeros_like(LiDAR_TeSet)

    X1_train = torch.tensor(trainset1).to(device)
    X2_train = torch.tensor(trainset2).to(device)
    X1_test = torch.tensor(valset1).to(device)
    X2_test = torch.tensor(valset2).to(device)

    val_dataset = random_mini_batches_standardtwoModality(X1_test, X2_test, Y_test)
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = True)
    if not with_dist:
        train_dataset = random_mini_batches_standardtwoModality(X1_train, X2_train, Y_train)
    else:
        if dist.get_rank() == 0:
            train_dataset = random_mini_batches_standardtwoModality(X1_train, torch.zeros_like(X2_train), Y_train)
        else:
            train_dataset = random_mini_batches_standardtwoModality(torch.zeros_like(X1_train), X2_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    return train_loader, val_loader

def train(model, optimizer, epochs, train_loader, test_loader, criterion):

    
    if with_dist is False:
        best_test_result = [float(0)]
    else:
        best_test_result = [float(0)] * dist.get_world_size()  # 每个rank单独维护
    

    for epoch in range(epochs+1):
        model.train() 

        num_correct = 0
        num_total = 0

        for batch_idx, (x1, x2, target) in enumerate(train_loader):
            x1 = x1.to(device, dtype=torch.float32)
            x2 = x2.to(device, dtype=torch.float32)
            target = torch.argmax(target, dim=1).to(device)
            
            outputs = model(x1, x2, with_svd=with_svd, k = svd_k) # 前向传播
            output1, _, _, _ = outputs

            loss = criterion(outputs, target)
            
            optimizer.zero_grad() 
            loss.backward()

            shared_layers = ['cross_a', 'cross_b','conv5', 'bn5', 'conv6', 'bn6', 'conv7']
            if with_dist is True:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                    if any(shared in name for shared in shared_layers):
                        dist.all_reduce(param.grad.data/2)
                    else:
                        dist.all_reduce(param.grad.data)
            optimizer.step() 
     
            pred_classes = torch.argmax(output1, dim=1)
            num_correct += torch.sum(pred_classes == target)
            num_total += target.size(0)
        train_acc = num_correct.item() / num_total
        train_loss = loss.item()
        scheduler.step()


        test_loss, test_acc, per_class_acc, test_aa, k = test(model, test_loader, criterion)
        if epoch % log_epoch == 0:
            if local_rank == 0:
                log_and_print("epoch %i: Train_loss: %f, Test_loss: %f, Train_OA: %f, Test_OA: %f, Test_AA: %f, Kappa:%.2f" % (epoch, train_loss, test_loss, train_acc, test_acc, test_aa, k))
                if save_excel is True:
                    save_result_excel(workbook, worksheet, excel_path, epoch, svd_k, test_acc, per_class_acc, test_aa, k)
            
    #     if local_rank == 0:
    #         test_tensor = torch.tensor([test_acc], device=device)
    #     else:
    #         test_tensor = torch.empty(1, device=device)
    #     if with_dist: dist.broadcast(test_tensor, src=0) 
        
    #     test_result = test_tensor.item()

    #     if test_result > best_test_result[local_rank]:
    #         best_test_result[local_rank] = test_result
    #         best_epoch = epoch
    #         torch.save(
    #             model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
    #             weight_path
    #         )

    # model.load_state_dict(torch.load(weight_path, weights_only=True))
    # test_loss, test_acc, per_class_acc, test_aa, k = test(model, test_loader, criterion)
    # if local_rank == 0:
    #     log_and_print("epoch %i: Train_loss: %f, Test_loss: %f, Train_OA: %f, Test_OA: %f, Test_AA: %f ,Kappa:%.2f" % (epoch, train_loss, test_loss, train_acc, test_acc, test_aa, k))
    #     if save_excel is True:
    #         save_result_excel(workbook, worksheet, excel_path, best_epoch, svd_k, test_acc, per_class_acc, test_aa, k)
    


def test(model, test_loader, criterion):
    model.eval() 
    test_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for (x1, x2, target) in test_loader:
            x1 = x1.to(device, dtype=torch.float32)
            x2 = x2.to(device, dtype=torch.float32)
            target = torch.argmax(target, dim=1).to(device)

            outputs = model(x1, x2, with_svd=with_svd)
            predictions, _, _, _= outputs

            loss = criterion(outputs, target)

            test_loss += loss.item() * target.size(0)
            all_targets.append(target)
            all_preds.append(torch.argmax(predictions, dim=1))

        all_targets = torch.cat(all_targets).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()

        c = confusion_matrix(all_targets, all_preds)
        test_acc, per_class_acc, aa, k = result_cal(torch.from_numpy(c))
        test_loss = test_loss / len(test_loader.dataset)
    return test_loss, test_acc, per_class_acc, aa, k


def inference_all(only_valid = False):
    model.eval() 
    ground_truth = scio.loadmat(os.path.join(dataset_dir, f'gt.mat'))['gt'].astype(np.int8) 
    try:
        HSI_MapSet = scio.loadmat(os.path.join(dataset_dir, f'HSI.mat'))['HSI']
        LiDAR_MapSet = scio.loadmat(os.path.join(dataset_dir, f'LiDAR.mat'))['LiDAR']
    except:
        HSI_MapSet = scio.loadmat(os.path.join(dataset_dir, f'data_HS_LR.mat'))['data_HS_LR']
        LiDAR_MapSet = scio.loadmat(os.path.join(dataset_dir, f'data_DSM.mat'))['data_DSM']

    HSI_MapSet, row, col, n_feature = mapset_normalization(mapset = HSI_MapSet, concate_pixel = concate_pixel, n_feature = hsi_n_feature, type = 'HSI')
    LiDAR_MapSet, row, col, n_feature = mapset_normalization(mapset = LiDAR_MapSet, concate_pixel = concate_pixel, n_feature = lidar_n_feature, type = 'LiDAR')

    drawall_idx = np.array([j for j, x in enumerate(ground_truth.reshape(row * col).ravel().tolist())])
    drawmap_loder1 = generate_batch(drawall_idx, HSI_MapSet, ground_truth.reshape(row * col), batch_size, patch_size, row, col, num_class = num_class, shuffle=False, only_valid = only_valid) 
    drawmap_loder2 = generate_batch(drawall_idx, LiDAR_MapSet, ground_truth.reshape(row * col), batch_size, patch_size, row, col, num_class = num_class, shuffle=False, only_valid = only_valid)


    model.eval()

    all_preds = []
    all_labels = []
    pred_test = np.full(row * col, min(np.unique(ground_truth)))  
    valid_idx = []
    total_processed = 0

    label = ground_truth.reshape(row * col)
    with torch.no_grad():
        for one, two in zip(drawmap_loder1, drawmap_loder2):
            data1 = one[0]
            data2 = two[0]
            valid_idx = two[1]
            if len(valid_idx) == 0:
                continue
            data1 = torch.tensor(data1)
            data2 = torch.tensor(data2)
            x1 = data1.to(torch.float32).to(device)
            x2 = data2.to(torch.float32).to(device)
            outputs = model(x1, x2, with_svd=with_svd)
            predictions, _, _, _ = outputs
            predictions = predictions.to(device)

            if len(predictions.shape) == 1:
                predictions = predictions.unsqueeze(0)
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            pred_classes = pred_classes + 1

            pred_test[valid_idx] = pred_classes

            all_preds.extend(pred_classes)
            all_labels.extend(label[valid_idx])
            total_processed += len(valid_idx)
    pred_test = pred_test.reshape(row, col) 
    print(f"Total processed data: {total_processed} (C*H*W)") 
    return pred_test, all_preds, all_labels



def output_visual(pred_test):
    log_and_print('Drawimg map')
    print(np.unique(pred_test))
    cmap = ListedColormap(colormap)
    plt.figure(figsize=(10, 8))
    plt.imshow(pred_test, cmap=cmap, interpolation='none', vmin=0, vmax=num_class)
    ticks = [0 if key == -1 else key for key in sorted(class_labels.keys())] 
    ticklabels = [class_labels[key] for key in sorted(class_labels.keys())] 
    cbar = plt.colorbar(ticks=ticks, label='Class Labels', boundaries=np.arange(-0.5, len(ticks) + 0.5))
    cbar.set_ticklabels(ticklabels)
    plt.savefig(map_fig_path, bbox_inches='tight', dpi=300) 


same_seeds(2) 

args = command_parser()
model_class = globals()[args.model_arch]
mode = args.mode
dataset_name = args.dataset_name.lower()
output_dir = args.output_dir
inference_only = bool(args.inference_only)
with_svd = bool(args.with_svd)
svd_k = args.svd_k if with_svd else 0 
save_excel = bool(args.save_excel)
save_png = bool(args.save_png)
log_epoch = args.log_epoch


# Setup logging
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
logging.basicConfig(filename=args.log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
# Log all input arguments
logging.info("Parsed Input Arguments:")
for arg, value in vars(args).items():
    log_and_print(f"{arg}: {value}")

with_dist = bool(f"{args.model_arch}" == "Fed_Fusion")
base_name = f"{args.model_arch}"
excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
map_fig_path = os.path.join(output_dir, f"{base_name}_{with_svd}_map_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png")
weight_folder = args.output_dir + '/weights/'
os.makedirs(weight_folder, exist_ok=True)


patch_size = 7
if dataset_name in dataset_config:
    config = dataset_config[dataset_name]
    dataset_dir = config['dataset_dir']
    hsi_n_feature = config['hsi_n_feature']
    lidar_n_feature = config['lidar_n_feature']
    concate_pixel = config['concate_pixel']
    class_labels = config['class_labels']
else:
    raise NameError(f"Dataset {dataset_name} not recognized")

cm_labels = list(class_labels.values())[1:]
num_class = max(class_labels.keys())
workbook, worksheet = create_excel(excel_path, mode, cm_labels)

palette = {0: (0, 0, 0)}
for k in range(len(cm_labels)):
    palette[k + 1] = tuple(custom_colors[k])
colormap = [np.array(color) / 255.0 for color in palette.values()]

local_rank = int(os.environ.get('RANK', 0))
if with_dist is True:
    if args.device == 'cpu':
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="gloo", init_method="env://", rank=local_rank)
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank) 
        dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank)
        device = torch.device("cuda", local_rank) 
else:
    device = torch.device(args.device)



model = model_class(hsi_n_feature, lidar_n_feature, num_class).to(device)

if with_svd:
    weight_path = os.path.join(weight_folder, f"{base_name}_svd_rank{local_rank}.pth")
else:
    weight_path = os.path.join(weight_folder, f"{base_name}_rank{local_rank}.pth")

    



batch_size = 64
epochs = 200
base_lr = 0.001
weight_decay = 0.001
optimizer = optim.Adam(model.parameters(), lr = base_lr, weight_decay = weight_decay)
scheduler = StepLR(optimizer, step_size = 30, gamma = 0.5)
log_and_print("epoches = {0}, batch size = {1}, base learning rate = {2}, weight decay = {3}".format(epochs, batch_size, base_lr, weight_decay))


criterion = Fed_Fusion_Loss(torch.ones(num_class).to(device)).to(device)

train_loader, test_loader = load_dataset(batch_size)

if inference_only is False:
    start_time = datetime.datetime.now()
    train(model, optimizer, epochs, train_loader, test_loader, criterion)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    log_and_print(f"Trining time: {elapsed_time.total_seconds():.2f} s")

pred_test, _, _= inference_all()

if local_rank == 0 and save_png is True:
    output_visual(pred_test)

if with_dist:
    dist.destroy_process_group()

