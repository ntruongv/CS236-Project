from vgg.utils import load_vgg16, vgg_preprocess
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
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


class LocalGraph(nn.Module):
    def __init__(self, image, cell_h=32, cell_gap=16, hidden_dim=256, out_dim=10):
        super(LocalGraph,self).__init__()
        self.vgg = load_vgg16('models')
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.trns = transforms.ToTensor()
        self.image = image
        width, height = image.size
        self.image_h = height
        self.image_w = width
        self.cell_h = cell_h
        self.cell_gap = cell_gap
        self.out_dim = int(256*self.cell_h/8*self.cell_h/8)
        self.fcn_1 = nn.Linear(self.out_dim, hidden_dim)
        self.fcn_2 = nn.Linear(hidden_dim, out_dim)

    def points_to_crop(self, pts):
        print("Start cropping")
        feat = [self.trns(self.image.crop((x[0]-self.cell_h/2, x[1]-self.cell_h/2, x[0]+self.cell_h/2, x[1]+self.cell_h/2))) for x in pts]
        print("Done cropping")
        feat = torch.stack(feat).cuda()
        feat = vgg_preprocess(feat)
        feat = self.vgg(feat)
        print(feat.shape)
        feat = F.relu(self.fcn_1(feat.view(-1,self.out_dim)), inplace=True)
        feat = F.relu(self.fcn_2(feat), inplace=True)
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
        _, dim_f = feat.size()
        fin_feat = torch.zeros((len(g_valid), dim_f))
        # fin_feat[g_valid] = feat
        count = 0
        for i, g_bool in enumerate(g_valid):
            if(g_bool):
                fin_feat[i] = feat[count]
                count +=1
        
        return fin_feat.view(n_batch, -1)#, c, h, w)
        

