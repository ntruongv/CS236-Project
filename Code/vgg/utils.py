# import torchfile
from torch.utils.data import DataLoader
from networks import PatchVgg16
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
import itertools
from PIL import Image

""" Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    
def load_vgg16(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.pth')):
            os.system('wget https://download.pytorch.org/models/vgg16-397923af.pth -O ' + os.path.join(model_dir, 'vgg16.pth'))
        vgglua = vgg16()
        vgglua.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.pth')))
        vgg = PatchVgg16()
        for (src, dst) in zip(vgglua.parameters(), vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = PatchVgg16()
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

class LocalGraph:
    def __init__(self, image):
        self.vgg = load_vgg16('models')
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.trns = transforms.ToTensor()
        self.image = image
        width, height = image.size
        self.image_h = height
        self.image_w = width
        self.cell_h = 24
        self.cell_gap = 16
    def points_to_crop(self, pts):
        feat = [self.trns(self.image.crop((x[0]-self.cell_h/2, x[1]-self.cell_h/2, x[0]+self.cell_h/2, x[1]+self.cell_h/2))) for x in pts]
        feat = torch.stack(feat)
        feat = vgg_preprocess(feat)
        feat = self.vgg(feat)
        return feat

    def extract_point(self, x, y):
        left, top = [],[]
        valid_l, valid_t = [], []
        for i in range(-1,2):
            left_i = x+i*self.cell_gap
            top_i = y+i*self.cell_gap
            if (self.cell_h/2) < left_i < (self.image_w-self.cell_h/2):
                left.append(left_i)
                valid_l.append(True)
            else:
                valid_l.append(False)
            if (self.cell_h/2) < top_i < (self.image_h-self.cell_h/2):
                top.append(top_i)
                valid_t.append(True)
            else:
                valid_t.append(False)
            
        cells = list(itertools.product(left,top))
        valid = [all(x) for x in itertools.product(valid_l,valid_t)]
        feat = self.points_to_crop(cells)
        _, c, h, w = feat.size()
        fin_feat = torch.zeros((len(valid), c, h, w))
        fin_feat[valid] = feat
        print(valid)
        return fin_feat
    
    def extract_batch(self, batch):
        n_batch = len(batch)
        g_cells, g_valid = [], []
        for pt in batch:
            x = pt[0]
            y = pt[1]
            left, top = [],[]
            valid_l, valid_t = [], []
            for i in range(-1,2):
                left_i = x+i*self.cell_gap
                top_i = y+i*self.cell_gap
                if (self.cell_h/2) < left_i < (self.image_w-self.cell_h/2):
                    left.append(left_i)
                    valid_l.append(True)
                else:
                    valid_l.append(False)
                if (self.cell_h/2) < top_i < (self.image_h-self.cell_h/2):
                    top.append(top_i)
                    valid_t.append(True)
                else:
                    valid_t.append(False)
                
            cells = list(itertools.product(left,top))
            valid = [all(x) for x in itertools.product(valid_l,valid_t)]
            g_cells.extend(cells)
            g_valid.extend(valid)
        feat = self.points_to_crop(g_cells)
        _, c, h, w = feat.size()
        fin_feat = torch.zeros((len(g_valid), c, h, w))
        fin_feat[g_valid] = feat
        return fin_feat.view(n_batch, -1, c, h, w)
        

