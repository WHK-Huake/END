import os
import cv2
import torch
import torch.nn as nn
import yaml
import losses
import random
from dataset.data import get_training_data, get_test_data
from models.END import Illum_YCRCB_Denoise_IN
from utils import network_parameters
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import utils
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import pandas as pd



# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Model
print('==> Build the model')

model_restored = Illum_YCRCB_Denoise_IN()

p_number = network_parameters(model_restored)
model_restored.cuda()

# Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

# Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)

result_dir = os.path.join(Train['SAVE_DIR'], mode, 'results')
utils.mkdir(result_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

# Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
train_log = os.path.join(log_dir, 'train.csv')
# 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['epoch', 'train Loss', 'val PSNR', 'val SSIM'])  # 列名
df.to_csv(train_log, mode='a', index=False)  # 路径可以根据需要更改

# Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    # path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    path_chk_rest = 'G:\low-light_image_enhancement\Our\END\checkpoints\MIT_END_v2\models\model_bestPSNR.pth'
    utils.load_checkpoint(model_restored, path_chk_rest)
    # start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    # utils.load_optim(optimizer, path_chk_rest)
    #
    # for i in range(1, start_epoch):
    #     scheduler.step()
    # new_lr = scheduler.get_lr()[0]
    # print('------------------------------------------------------------------')
    # print("==> Resuming Training with learning rate:", new_lr)
    # print('------------------------------------------------------------------')

## Loss
# PSNR_loss = losses.PSNRLoss()
char_criterion = losses.CharbonnierLoss()
ssim_criterion = losses.SSIM_Loss()
denoise_criterion = nn.MSELoss()

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=0, drop_last=False)
val_dataset = get_test_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'] + 1)}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(0)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_epoch_psnr = 0
total_start_time = time.time()

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    model_restored.train()

    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        optimizer.zero_grad()

        target_y, target_uv = target[:, 0, :, :].unsqueeze(1), target[:, 1:3, :, :]
        restored_y, restored_uv, restored = model_restored(input_)

        # Compute loss
        loss_y = char_criterion(restored_y, target_y) + ssim_criterion(restored_y, target_y)
        loss_uv = char_criterion(restored_uv, target_uv) + ssim_criterion(restored_uv, target_uv)
        loss_yuv = char_criterion(restored, target) + ssim_criterion(restored, target)

        loss = 0.01*loss_y + 0.1*loss_uv + loss_yuv

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            filename = data_val[2]
            with torch.no_grad():
                restored_y, restored_uv, restored = model_restored(input_)

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(0)
            restored = cv2.cvtColor(restored, cv2.COLOR_YCrCb2BGR)

            target = target.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(0)
            target = cv2.cvtColor(target, cv2.COLOR_YCrCb2BGR)

            psnr_val_rgb.append(utils.PSNR(target, restored))

        psnr_val_rgb = np.mean(psnr_val_rgb)

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # 将数据保存在一维列表
        list = [epoch, epoch_loss, psnr_val_rgb]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        data.to_csv(train_log, mode='a', header=False, index=False) # mode设为a,就可以向csv文件追加数据了
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
