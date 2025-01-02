import os
import argparse
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import utils
from dataset.data import get_unpair_data
from models.END import Illum_YCRCB_Denoise_IN


parser = argparse.ArgumentParser(description='END')
parser.add_argument('--input_dir', default=r'G:\low-light_image_enhancement\Our\dataset\test_datasets', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/MIT-END', type=str, help='Directory for results')
parser.add_argument('--weights', default=r'G:\low-light_image_enhancement\Github\END\checkpoints', type=str,
                    help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = Illum_YCRCB_Denoise_IN().cuda()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration.eval()

datasets = ['LIME', 'NPE', 'DICM', 'MEF']     #'LIME', 'NPE', 'DICM', 'MEF'

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = get_unpair_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        psnr_yuv = []
        ssim_yuv = []

        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            input_ = data_test[0].cuda()
            filenames = data_test[1]
            print(filenames)
            with torch.no_grad():
                x1, x2, restored = model_restoration(input_)

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(0)
            restored = cv2.cvtColor(restored, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite((os.path.join(result_dir, filenames[0] + '.png')), restored)
