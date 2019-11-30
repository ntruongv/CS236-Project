# import torchfile
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.models import vgg16
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time

""" Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    
def load_vgg16(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.pth')):
            os.system('wget https://download.pytorch.org/models/vgg16-397923af.pth -O ' + os.path.join(model_dir, 'vgg16.pth'))
        vgglua = vgg16()
        vgglua.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.pth')))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters(), vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch