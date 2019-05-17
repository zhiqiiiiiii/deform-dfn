import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time
from pprint import pprint
from torch.autograd import Variable
from torch import optim
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from data.BouncingMnistDataset import BouncingMnistDataset
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
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--dy_filter_size', type=int, default=3)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='deform_dfn', help='deform_dfn|dfn')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--loss', type=str, default='BCE', help='BCE|MSE')
parser.add_argument('--display_freq', type=int, default=200, help='frequency to display result')
parser.add_argument('--save_freq', type=int, default=500, help='frequency to save results')
parser.add_argument('--epoch_save_freq', type=int, default=5, help='frequency to save model')
param = parser.parse_args()

device = torch.device('cpu')
# save checkpoints
check_dir = param.checkpoint_path
save_dir = os.path.join(check_dir, param.name)
img_dir = os.path.join(save_dir, 'imgs')
mkdir(check_dir)
mkdir(save_dir)
mkdir(img_dir)
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


mnistDataset='./mnist.h5'
try:
    f = h5py.File(mnistDataset, 'r')
    dataset = BouncingMnistDataset(f)
    f.close()
except:
    print('Please set the correct path to MNIST dataset')
    sys.exit()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size, num_workers=0)

if param.model == 'deform_dfn':
    net = Deform_DFN(1, param.dy_filter_size)
elif param.model == 'dfn':
    net = DFN(1, param.dy_filter_size)

if param.use_cuda and torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda')
    net = net.cuda()
if param.continue_train:
    load_path = '%s_%s.pth' % (param.name, param.model_load)
    load_path = os.path.join(save_dir, load_path)
    state_dict = torch.load(load_path, map_location=device)
    net.load_state_dict(state_dict)

optimizer = optim.Adam(net.parameters(), lr=param.learning_rate)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
if param.loss == 'BCE':
    criterion = BCE_loss()
elif param.loss == 'MSE':
    criterion = nn.MSELoss()

net.train()
unloader = torchvision.transforms.ToPILImage()

iter = 0
for epoch in range(param.epoch_start, param.epoch_start+param.epochs):
    
    for i, data in enumerate(data_loader):
        
        if param.use_cuda and torch.cuda.is_available():
            data = data.cuda()
            
        optimizer.zero_grad()
        predictions = net(data,num_output_frames=param.num_output_frames,use_cuda=param.use_cuda)
        data = torch.unbind(data, dim=-1)
        true_frames = data[param.num_input_frames:param.num_input_frames+param.num_output_frames]
        loss = 0
        for j in range(param.num_output_frames):
            loss += criterion(predictions[j], true_frames[j])
        loss = loss/param.num_output_frames

        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        if iter%param.display_freq == 0:
            message = 'epoch:{}, iter:{} --- loss: {}'.format(epoch, i, loss.item())
            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        if iter%param.save_freq == 0:
            for k in range(param.num_output_frames):
                img_pred = predictions[k][0,:,:,:]
                img_pred = unloader(img_pred.detach().cpu())
                img_filename = ('%s_pred_%s.png' % (epoch, k))
                path = os.path.join(img_dir, img_filename)
                img_pred.save(path)

                img_true = true_frames[k][0,:,:,:].clone()
                img_true = unloader(img_true.detach().cpu())
                img_filename = ('%s_true_%s.png' % (epoch, k))
                path = os.path.join(img_dir, img_filename)
                img_true.save(path)
            for k in range(param.num_input_frames):
                img = data[k][0,:,:,:]
                img = unloader(img.detach().cpu())
                img_filename = ('%s_input_frame_%s.png' % (epoch, k))
                path = os.path.join(img_dir, img_filename)
                img.save(path)
            
        iter += 1
        
    if epoch % param.epoch_save_freq == 0:
        message = 'saving the model at the end of epoch %d, total iters %d' % (epoch, iter)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)
        path = os.path.join(save_dir, ('%s_%s.pth' % (param.name, epoch)))
        torch.save(net.state_dict(), path)
        path = os.path.join(save_dir, ('%s_latest.pth' % param.name))
        torch.save(net.state_dict(), path)
