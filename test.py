import os
from math import log10
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time
from pprint import pprint
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from data.BouncingMovingMnistTest import BouncingMnistTestDataset
import torchvision
from models.networks import BCE_loss, DFN, Deform_DFN
import sys
import h5py
import os

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--num_input_frames', type=int, default=10)
parser.add_argument('--num_output_frames', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--dy_filter_size', type=int, default=3)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='deform_dfn', help='deform_dfn|dfn')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
param = parser.parse_args()

device = torch.device('cpu')
# save checkpoints
mkdir(param.result_path)
check_dir = os.path.join(param.checkpoint_path, param.name)
save_dir = os.path.join(param.result_path, param.name)
mkdir(save_dir)
# save log file
log_name = os.path.join(save_dir, 'log.txt')
message = ''
message += '----------------- Options ---------------\n'
for k, v in sorted(vars(param).items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
        comment = '\t[default: %s]' % str(default)
    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
print(message)
with open(log_name, "a") as log_file:
    log_file.write('%s\n' % message)

dataset = BouncingMnistTestDataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size, num_workers=0, shuffle=False)

if param.model == 'deform_dfn':
    net = Deform_DFN(1, param.dy_filter_size)
elif param.model == 'dfn':
    net = DFN(1, param.dy_filter_size)

if param.use_cuda and torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda')
    net = net.cuda()

load_path = '%s_%s.pth' % (param.name, param.model_load)
load_path = os.path.join(check_dir, load_path)
state_dict = torch.load(load_path, map_location=device)
net.load_state_dict(state_dict)

net.eval()
unloader = torchvision.transforms.ToPILImage()

criterionMSE = nn.MSELoss()
psnr_avg = 0
mse_avg = 0
ssim_avg = 0
for i, data in enumerate(data_loader):

    if param.use_cuda and torch.cuda.is_available():
        data = data.cuda()

    predictions = net(data,num_output_frames=param.num_output_frames,use_cuda=param.use_cuda)
    data = torch.unbind(data, dim=-1)
    true_frames = data[param.num_input_frames:param.num_input_frames+param.num_output_frames]
    
#     message = 'epoch:{}, iter:{} --- loss: {}'.format(epoch, i, loss.item())
#     print(message)
#     with open(log_name, "a") as log_file:
#         log_file.write('%s\n' % message)

    for k in range(param.num_output_frames):
        img_pred = predictions[k][0,:,:,:]
        img_pred = img_pred.detach().cpu()
        img_true = true_frames[k][0,:,:,:].clone()
        img_true = img_true.detach().cpu()
        
        mse = criterionMSE(img_pred, img_true)
        psnr = 10 * log10(1 / mse.item())
        message = 'Sample %s, predicting %s frame, mse: %.3f, psnr: %.3f' % (i, k, mse, psnr)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)
        
        mse_avg += mse
        psnr_avg += psnr
        
        img_pred = unloader(img_pred)
        img_filename = ('%s_pred_%s.png' % (i, k))
        path = os.path.join(save_dir, img_filename)
        img_pred.save(path)
        img_true = unloader(img_true)
        img_filename = ('%s_true_%s.png' % (i, k))
        path = os.path.join(save_dir, img_filename)
        img_true.save(path)
        
    for k in range(param.num_input_frames):
        img = data[k][0,:,:,:]
        img = unloader(img.detach().cpu())
        img_filename = ('%s_input_frame_%s.png' % (i, k))
        path = os.path.join(save_dir, img_filename)
        img.save(path)
        #print('saving %s_input_frame_%s.png' % (i,k))
        
mse_avg = mse_avg/(param.num_output_frames*len(dataset))
psnr_avg = psnr_avg/(param.num_output_frames*len(dataset))
message = 'Average mse: %.3f, Average psnr: %.3f' % (mse_avg, psnr_avg)
print(message)
with open(log_name, "a") as log_file:
    log_file.write('%s\n' % message)