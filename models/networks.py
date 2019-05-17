import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import optim
from torch.autograd import Function, Variable
from models.deformconv2d import DeformConv2D

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class single_deconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(single_deconv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.en1 = single_conv(in_ch, 32, kernel_size=9, stride=2, padding=9//2)
        self.en2 = single_conv(32, 64, kernel_size=9, stride=2, padding=9//2)
        self.en3 = single_conv(64, 128, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        return x
    
class Encoder2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder2, self).__init__()
        self.en1 = single_conv(in_ch, 32, kernel_size=9, stride=2, padding=9//2)
        self.en2 = single_conv(32, 64, kernel_size=9, stride=2, padding=9//2)
        self.en3 = single_conv(64, 128, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        return x
    
class Decoder2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder2, self).__init__()
        self.de1 = single_deconv(128, 64, kernel_size=3, stride=2, padding=1)
        self.de2 = single_deconv(64, 32, kernel_size=3, stride=2, padding=1)
        self.de3 = single_deconv(32, out_ch, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = F.interpolate(x, (64, 64))
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.de1 = single_deconv(128, 64, kernel_size=3, stride=2, padding=1)
        self.de2 = single_deconv(64, 32, kernel_size=3, stride=2, padding=1)
        self.de3 = single_deconv(32, out_ch, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = F.interpolate(x, (64, 64))
        return x
        
class EncoderDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderDecoder, self).__init__()
        self.en1 = single_conv(in_ch, 32, kernel_size=9, stride=2, padding=9//2)
        self.en2 = single_conv(32, 64, kernel_size=9, stride=2, padding=9//2)
        self.en3 = single_conv(64, 128, kernel_size=9, stride=2, padding=9//2)
        self.de1 = single_deconv(128, 64, kernel_size=9, stride=2, padding=9//2)
        self.de2 = single_deconv(64, 32, kernel_size=9, stride=2, padding=9//2)
        self.de3 = single_deconv(32, out_ch, kernel_size=9, stride=2, padding=9//2)
    
    def forward(self, x):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = F.interpolate(x, (64, 64))
        return x
    
class BCE_loss(nn.Module):
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, input, target):        
        return self.bceloss(input, target)

    def bceloss(self, input, target):
        eps = 0.00001
        c_pred = torch.clamp(input, eps, 1-eps)
        bce = -target*torch.log(c_pred) - (1-target)*torch.log(1-c_pred)
        frame_loss = torch.mean(bce, (1,2,3))
        batch_loss = torch.mean(frame_loss, -1)
        return torch.sum(batch_loss)
    
class deform_DFN(nn.Module):
    def __init__(self, in_ch, dy_filter_size=(3,3)):
        super(deform_DFN, self).__init__()
        self._filter_size = dy_filter_size
        self.filter = dy_filter_size[0]*dy_filter_size[1]
        self.encoder_decoder = EncoderDecoder(in_ch, 3*dy_filter_size[0]*dy_filter_size[1])
        
        self.hidden_conv = nn.Sequential(
            nn.Conv2d(3*dy_filter_size[0]*dy_filter_size[1], 9, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, in_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_ch)
        )
        
        self.deform = DeformConv2D(1, 1)
        self.unfold = nn.Unfold(kernel_size=(3,3), padding=1)
        self.unfold_deform = nn.Unfold(kernel_size=(3,3), padding=1, stride=3)
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10):
        hidden_state = torch.zeros((batch_size, 128, 8, 8), dtype=torch.float32)
        input_frames = torch.unbind(input_frames, dim=-1)
        output_frames = []
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = input_frames[i]
            else:
                input_frame = prediction
            prediction, dynamic_filters, hidden_state = self.predict(input_frame, hidden_state)
            
            if i >= num_input_frames - 1:
                output_frames.append(prediction)
        return torch.stack(output_frames, dim=-1)
    
        
    def predict(self, input_frame, hidden_state):
        input = input_frame + hidden_state
        hidden = self.encoder_decoder(input)
        hidden_state = self.hidden_conv(hidden) + hidden_state
        dynamic_filters = F.softmax(hidden)
        dynamic_filters_conv = dynamic_filters[:,0:self.filter,:,:]
        dynamic_filters_offset_x = dynamic_filters[:,self.filter:self.filter*2,:,:]
        dynamic_filters_offset_y = dynamic_filters[:,self.filter*2:,:,:]
       
        dynamic_filters_offset_x = dynamic_filters_offset_x.view(dynamic_filters_offset_x.shape[0],dynamic_filters_offset_x.shape[1],64*64)
        dynamic_filters_offset_y = dynamic_filters_offset_y.view(dynamic_filters_offset_y.shape[0],dynamic_filters_offset_y.shape[1],64*64)
        dynamic_filters_conv = dynamic_filters_conv.view(dynamic_filters_conv.shape[0],dynamic_filters_conv.shape[1],64*64)
        
        input_frame_unfold = self.unfold(input_frame)
        offsets_x = input_frame_unfold*dynamic_filters_offset_x
        offsets_x = offsets_x.view(offsets_x.shape[0], offsets_x.shape[1], 64, 64)
        offsets_y = input_frame_unfold*dynamic_filters_offset_y
        offsets_y = offsets_y.view(offsets_y.shape[0], offsets_y.shape[1], 64, 64)
        offset = torch.cat((offsets_x,offsets_y),1)
        x_offset = self.deform(input_frame, offset)
        blocks = self.unfold_deform(x_offset)
        
        prediction = blocks*dynamic_filters_conv
        prediction = torch.sum(prediction, dim=1)
        prediction = prediction.view(-1, 64, 64)
        prediction = prediction.unsqueeze(1)
        
        return prediction, dynamic_filters, hidden_state
    
    
class DFN1(nn.Module):
    def __init__(self, in_ch, dy_filter_size=(9,9)):
        super(DFN1, self).__init__()
        self._filter_size = dy_filter_size
        self.encoder = Encoder(in_ch, dy_filter_size[0]*dy_filter_size[1])
        self.hidden_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.decoder = Decoder(in_ch, dy_filter_size[0]*dy_filter_size[1])
        self.unfold = nn.Unfold(kernel_size=dy_filter_size, padding=dy_filter_size[0]//2)
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10):
        hidden_state = torch.zeros((input_frames.shape[0], 128, 8, 8), dtype=torch.float32)
        if torch.cuda.is_available and use_cuda:
            hidden_state = hidden_state.cuda()
        input_frames = torch.unbind(input_frames, dim=-1)
        output_frames = []
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = input_frames[i]
            else:
                input_frame = prediction
            output, hidden_state = self.predict(input_frame, hidden_state)
            if i >= num_input_frames - 1:
                dynamic_filters = F.softmax(output)
                dynamic_filters = dynamic_filters.view(dynamic_filters.shape[0],dynamic_filters.shape[1],64*64)
                input_frame_unfold = self.unfold(input_frame)
                output_frame = input_frame_unfold*dynamic_filters
                prediction = torch.sum(output_frame, dim=1)
                prediction = prediction.view(-1, 64, 64)
                prediction = prediction.unsqueeze(1)
                output_frames.append(prediction)
        #return torch.stack(output_frames, dim=-1)
        return output_frames
    
        
    def predict(self, input_frame, hidden_state):
        hidden = self.encoder(input_frame)
        hidden_state = self.hidden_conv(hidden) + hidden_state
        output = self.decoder(hidden_state)
        return output, hidden_state
    
    
class DeformDFN1(nn.Module):
    def __init__(self, in_ch, dy_filter_size=(9,9)):
        super(DeformDFN1, self).__init__()
        self._filter_size = dy_filter_size
        self.encoder = Encoder(in_ch, dy_filter_size[0]*dy_filter_size[1])
        self.hidden_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.decoder = Decoder(in_ch, 3*dy_filter_size[0]*dy_filter_size[1])
        self.unfold = nn.Unfold(kernel_size=dy_filter_size, padding=dy_filter_size[0]//2)
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10):
        hidden_state = torch.zeros((input_frames.shape[0], 128, 8, 8), dtype=torch.float32)
        if torch.cuda.is_available and use_cuda:
            hidden_state = hidden_state.cuda()
        input_frames = torch.unbind(input_frames, dim=-1)
        output_frames = []
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = input_frames[i]
            else:
                input_frame = prediction
            output, hidden_state = self.predict(input_frame, hidden_state)
            if i >= num_input_frames - 1:
                dynamic_filters = F.softmax(output)
                dynamic_filters = dynamic_filters.view(dynamic_filters.shape[0],dynamic_filters.shape[1],64*64)
                input_frame_unfold = self.unfold(input_frame)
                
                
                dynamic_filters_conv = dynamic_filters[:,0:self.filter,:,:]
                dynamic_filters_offset_x = dynamic_filters[:,self.filter:self.filter*2,:,:]
                dynamic_filters_offset_y = dynamic_filters[:,self.filter*2:,:,:]
                dynamic_filters_offset_x = dynamic_filters_offset_x.view(dynamic_filters_offset_x.shape[0],dynamic_filters_offset_x.shape[1],64*64)
                dynamic_filters_offset_y = dynamic_filters_offset_y.view(dynamic_filters_offset_y.shape[0],dynamic_filters_offset_y.shape[1],64*64)
                dynamic_filters_conv = dynamic_filters_conv.view(dynamic_filters_conv.shape[0],dynamic_filters_conv.shape[1],64*64)
        
       
                offsets_x = input_frame_unfold*dynamic_filters_offset_x
                offsets_x = offsets_x.view(offsets_x.shape[0], offsets_x.shape[1], 64, 64)
                offsets_y = input_frame_unfold*dynamic_filters_offset_y
                offsets_y = offsets_y.view(offsets_y.shape[0], offsets_y.shape[1], 64, 64)
                offset = torch.cat((offsets_x,offsets_y),1)
                x_offset = self.deform(input_frame, offset)
                blocks = self.unfold_deform(x_offset)
        
                prediction = blocks*dynamic_filters_conv
                prediction = torch.sum(prediction, dim=1)
                prediction = prediction.view(-1, 64, 64)
                prediction = prediction.unsqueeze(1)
                
                output_frames.append(prediction)
        #return torch.stack(output_frames, dim=-1)
        return output_frames
    
        
    def predict(self, input_frame, hidden_state):
        hidden = self.encoder(input_frame)
        hidden_state = self.hidden_conv(hidden) + hidden_state
        output = self.decoder(hidden_state)
        return output, hidden_state
    
class DFN2(nn.Module):
    def __init__(self, in_ch, dy_filter_size=(9,9)):
        super(DFN2, self).__init__()
        self._filter_size = dy_filter_size
        self.encoder = Encoder2(in_ch, dy_filter_size[0]*dy_filter_size[1])
        self.hidden_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.decoder = Decoder2(in_ch, dy_filter_size[0]*dy_filter_size[1])
        self.unfold = nn.Unfold(kernel_size=dy_filter_size, padding=dy_filter_size[0]//2)
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10, use_cuda=False):
        hidden_state = torch.zeros((input_frames.shape[0], 128, 16, 16), dtype=torch.float32)
        if torch.cuda.is_available and use_cuda:
            hidden_state = hidden_state.cuda()
        input_frames = torch.unbind(input_frames, dim=-1)
        output_frames = []
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = input_frames[i]
            else:
                input_frame = prediction
            output, hidden_state = self.predict(input_frame, hidden_state)
            if i >= num_input_frames - 1:
                dynamic_filters = F.softmax(output)
                dynamic_filters = dynamic_filters.view(dynamic_filters.shape[0],dynamic_filters.shape[1],64*64)
                input_frame_unfold = self.unfold(input_frame)
                output_frame = input_frame_unfold*dynamic_filters
                prediction = torch.sum(output_frame, dim=1)
                prediction = prediction.view(-1, 64, 64)
                prediction = prediction.unsqueeze(1)
                output_frames.append(prediction)
        #return torch.stack(output_frames, dim=-1)
        return output_frames
    
    def predict(self, input_frame, hidden_state):
        hidden = self.encoder(input_frame)
        hidden_state = self.hidden_conv(hidden) + hidden_state
        output = self.decoder(hidden_state)
        return output, hidden_state
    
    
class DyNet(nn.Module):
    def __init__(self, in_ch, dy_filter_size=9):
        super(DyNet, self).__init__()
        self._filter_size = dy_filter_size
        self.filter = dy_filter_size**2
        
        self.en1 = single_conv(in_ch, 32)
        self.en2 = single_conv(32, 32, stride=2)
        self.en3 = single_conv(32, 64)
        self.en4 = single_conv(64, 64, stride=2)
        self.en5 = single_conv(64, 64)
        
        self.mid1 = single_conv(64, 128)
        self.mid2 = single_conv(128, 128)
        self.mid_h1 = single_conv(128, 128)
        self.mid_h2 = single_conv(128, 128)
        
        self.de1 = single_conv(128, 64)
        self.de2 = single_conv(64, 64)
        self.de_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de3 = single_conv(64, 64)
        self.de_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de4 = single_conv(64, 64)
        self.de5 = single_conv(64, 128, kernel_size=1, padding=0)
        
        self.dyf = nn.Conv2d(128, 3*self.filter, kernel_size=1, stride=1, padding=0)
        
        self.unfold = nn.Unfold(kernel_size=self._filter_size, padding=self._filter_size//2)
        self.deform = DeformConv2D(1, 1, kernel_size=self._filter_size)
        self.unfold_deform = nn.Unfold(kernel_size=(self._filter_size, self._filter_size), padding=self._filter_size//2, stride=self._filter_size)
        
        
    def _build(self, input_frames, use_cuda=False):
        batch_size = input_frames.shape[0]
        self.hidden_state = torch.zeros((batch_size, 128, 32, 32), dtype=torch.float32)
        if use_cuda and torch.cuda.is_available:
            self.hidden_state = self.hidden_state.cuda()
            input_frames = input_frames.cuda()
        self.input_frames = torch.unbind(input_frames, dim=-1)
        self.output_frames = []
            
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10, use_cuda=False):
        self._build(input_frames, use_cuda)
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = self.input_frames[i]
            else:
                input_frame = prediction
            dynamic_filters, self.hidden_state = self.predict(input_frame, self.hidden_state)
            
            if i >= num_input_frames - 1:
                input_frame_unfold = self.unfold(input_frame)
                
                dynamic_filters_conv = dynamic_filters[:,0:self.filter,:,:]
                dynamic_filters_offset_x = dynamic_filters[:,self.filter:self.filter*2,:,:]
                dynamic_filters_offset_y = dynamic_filters[:,self.filter*2:,:,:]
                dynamic_filters_offset_x = dynamic_filters_offset_x.view(dynamic_filters_offset_x.shape[0],dynamic_filters_offset_x.shape[1],64*64)
                dynamic_filters_offset_y = dynamic_filters_offset_y.view(dynamic_filters_offset_y.shape[0],dynamic_filters_offset_y.shape[1],64*64)
                dynamic_filters_conv = dynamic_filters_conv.view(dynamic_filters_conv.shape[0],dynamic_filters_conv.shape[1],64*64)
        
       
                offsets_x = input_frame_unfold*dynamic_filters_offset_x
                offsets_x = offsets_x.view(offsets_x.shape[0], offsets_x.shape[1], 64, 64)
                offsets_y = input_frame_unfold*dynamic_filters_offset_y
                offsets_y = offsets_y.view(offsets_y.shape[0], offsets_y.shape[1], 64, 64)
                offset = torch.cat((offsets_x,offsets_y),1)
                x_offset = self.deform(input_frame, offset)
                blocks = self.unfold_deform(x_offset)
        
                prediction = blocks*dynamic_filters_conv
                prediction = torch.sum(prediction, dim=1)
                prediction = prediction.view(-1, 64, 64)
                prediction = prediction.unsqueeze(1)
                
                self.output_frames.append(prediction)
        return self.output_frames
    
    def encoder(self, input_frame):
        out = self.en1(input_frame)
        out = self.en2(out)
        out = self.en3(out)
        out = self.en5(out)
        return out
        
    def mid(self, hidden, hidden_state):
        out = self.mid1(hidden)
        hid = self.mid_h1(hidden_state)
        hid = self.mid_h2(hid)
        out = out + hid
        return out
    
    def decoder(self, hidden_state):
        out = self.de1(hidden_state)
        out = self.de2(out)
        out = self.de_up1(out)
        out = self.de3(out)
        out = self.de4(out)
        out = self.de5(out)
        return out
        
    def predict(self, input_frame, hidden_state):
        hidden = self.encoder(input_frame)
        self.hidden_state = self.mid(hidden, hidden_state)
        output = self.decoder(self.hidden_state)
        dynamic_filters = self.dyf(output)
        dynamic_filters = F.softmax(dynamic_filters)
        return dynamic_filters, self.hidden_state
    
class DFN(nn.Module):
    def __init__(self, in_ch, dy_filter_size=9):
        super(DFN, self).__init__()
        self._filter_size = dy_filter_size
        self.filter = dy_filter_size**2
        
        self.en1 = single_conv(in_ch, 32)
        self.en2 = single_conv(32, 32, stride=2)
        self.en3 = single_conv(32, 64)
        self.en4 = single_conv(64, 64, stride=2)
        self.en5 = single_conv(64, 64)
        
        self.mid1 = single_conv(64, 128)
        self.mid2 = single_conv(128, 128)
        self.mid_h1 = single_conv(128, 128)
        self.mid_h2 = single_conv(128, 128)
        
        self.de1 = single_conv(128, 64)
        self.de2 = single_conv(64, 64)
        self.de_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de3 = single_conv(64, 64)
        self.de_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de4 = single_conv(64, 64)
        self.de5 = single_conv(64, 128, kernel_size=1, padding=0)
        
        self.dyf = nn.Conv2d(128, self.filter, kernel_size=1, stride=1, padding=0)
        self.unfold = nn.Unfold(kernel_size=self._filter_size, padding=self._filter_size//2)  
        
    def _build(self, input_frames, use_cuda=False):
        batch_size = input_frames.shape[0]
        self.hidden_state = torch.zeros((batch_size, 128, 32, 32), dtype=torch.float32)
        if use_cuda and torch.cuda.is_available:
            self.hidden_state = self.hidden_state.cuda()
            input_frames = input_frames.cuda()
        self.input_frames = torch.unbind(input_frames, dim=-1)
        self.output_frames = []
            
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10, use_cuda=False):
        self._build(input_frames, use_cuda)
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = self.input_frames[i]
            else:
                input_frame = prediction
            dynamic_filters, self.hidden_state = self.predict(input_frame, self.hidden_state)
            
            if i >= num_input_frames - 1:
                input_frame_unfold = self.unfold(input_frame)
                dynamic_filters= dynamic_filters.view(dynamic_filters.shape[0],dynamic_filters.shape[1],64*64)
                prediction = input_frame_unfold*dynamic_filters
                prediction = torch.sum(prediction, dim=1)
                prediction = prediction.view(-1, 64, 64)
                prediction = prediction.unsqueeze(1)
                
                self.output_frames.append(prediction)
        return self.output_frames
    
    def encoder(self, input_frame):
        out = self.en1(input_frame)
        out = self.en2(out)
        out = self.en3(out)
        out = self.en5(out)
        return out
        
    def mid(self, hidden, hidden_state):
        out = self.mid1(hidden)
        hid = self.mid_h1(hidden_state)
        hid = self.mid_h2(hid)
        out = out + hid
        return out
    
    def decoder(self, hidden_state):
        out = self.de1(hidden_state)
        out = self.de2(out)
        out = self.de_up1(out)
        out = self.de3(out)
        out = self.de4(out)
        out = self.de5(out)
        return out
        
    def predict(self, input_frame, hidden_state):
        hidden = self.encoder(input_frame)
        self.hidden_state = self.mid(hidden, hidden_state)
        output = self.decoder(self.hidden_state)
        dynamic_filters = self.dyf(output)
        dynamic_filters = F.softmax(dynamic_filters)
        return dynamic_filters, self.hidden_state

    
class Deform_DFN(nn.Module):
    def __init__(self, in_ch, dy_filter_size=9):
        super(Deform_DFN, self).__init__()
        self._filter_size = dy_filter_size
        self.filter = dy_filter_size**2
        
        self.en1 = single_conv(in_ch, 32)
        self.en2 = single_conv(32, 32, stride=2)
        self.en3 = single_conv(32, 64)
        self.en4 = single_conv(64, 64, stride=2)
        self.en5 = single_conv(64, 64)
        
        self.mid1 = single_conv(64, 128)
        self.mid2 = single_conv(128, 128)
        self.mid_h1 = single_conv(128, 128)
        self.mid_h2 = single_conv(128, 128)
        
        self.de1 = single_conv(128, 64)
        self.de2 = single_conv(64, 64)
        self.de_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de3 = single_conv(64, 64)
        self.de_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de4 = single_conv(64, 64)
        self.de5 = single_conv(64, 128, kernel_size=1, padding=0)
        
        self.dyf = nn.Conv2d(128, self.filter, kernel_size=1, stride=1, padding=0)
        
        self.de1_2 = single_conv(128, 64)
        self.de2_2 = single_conv(64, 64)
        self.de_up1_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de3_2 = single_conv(64, 64)
        self.de_up2_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.de4_2 = single_conv(64, 64)
        self.de5_2 = single_conv(64, 128, kernel_size=1, padding=0)
        
        self.dyf_2 = nn.Conv2d(128, 2*self.filter, kernel_size=1, stride=1, padding=0)
        
        self.unfold = nn.Unfold(kernel_size=self._filter_size, padding=self._filter_size//2)
        self.deform = DeformConv2D(1, 1, kernel_size=self._filter_size)
        self.unfold_deform = nn.Unfold(kernel_size=(self._filter_size, self._filter_size), padding=self._filter_size//2, stride=self._filter_size)        
        
    def _build(self, input_frames, use_cuda=False):
        batch_size = input_frames.shape[0]
        self.hidden_state = torch.zeros((batch_size, 128, 32, 32), dtype=torch.float32)
        if use_cuda and torch.cuda.is_available:
            self.hidden_state = self.hidden_state.cuda()
            input_frames = input_frames.cuda()
        self.input_frames = torch.unbind(input_frames, dim=-1)
        self.output_frames = []
        self.offset = []
        
    def forward(self, input_frames, num_input_frames=10, num_output_frames=10, use_cuda=False):
        self._build(input_frames, use_cuda)
        for i in range(num_input_frames + num_output_frames - 1):
            if i < num_input_frames:
                input_frame = self.input_frames[i]
            else:
                input_frame = prediction
            dynamic_filters, offset, self.hidden_state = self.predict(input_frame, self.hidden_state)
            
            if i >= num_input_frames - 1:
                self.offset.append(offset)
                input_frame_unfold = self.unfold(input_frame)
                
                dynamic_filters = dynamic_filters.view(dynamic_filters.shape[0],dynamic_filters.shape[1],64*64)
        
                x_offset = self.deform(input_frame, offset)
                blocks = self.unfold_deform(x_offset)
        
                prediction = blocks*dynamic_filters
                prediction = torch.sum(prediction, dim=1)
                prediction = prediction.view(-1, 64, 64)
                prediction = prediction.unsqueeze(1)
                
                self.output_frames.append(prediction)
        return self.output_frames
    
    def encoder(self, input_frame):
        out = self.en1(input_frame)
        out = self.en2(out)
        out = self.en3(out)
        out = self.en5(out)
        return out
        
    def mid(self, hidden, hidden_state):
        out = self.mid1(hidden)
        hid = self.mid_h1(hidden_state)
        hid = self.mid_h2(hid)
        out = out + hid
        return out
    
    def decoder(self, hidden_state):
        out = self.de1(hidden_state)
        out = self.de2(out)
        out = self.de_up1(out)
        out = self.de3(out)
        out = self.de4(out)
        out = self.de5(out)
        return out
    
    def decoder_2(self, hidden_state):
        out = self.de1_2(hidden_state)
        out = self.de2_2(out)
        out = self.de_up1_2(out)
        out = self.de3_2(out)
        out = self.de4_2(out)
        out = self.de5_2(out)
        return out   
        
    def predict(self, input_frame, hidden_state):
        hidden = self.encoder(input_frame)
        self.hidden_state = self.mid(hidden, hidden_state)
        output = self.decoder(self.hidden_state)
        dynamic_filters = self.dyf(output)
        dynamic_filters = F.softmax(dynamic_filters)
        output_2 = self.decoder_2(self.hidden_state)
        offset = self.dyf_2(output_2)
        return dynamic_filters, offset, self.hidden_state
    
    def get_offset(self):
        return self.offset