from utils import vgg_preprocess, load_vgg16
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms

# vgg_model_path = ""
vgg = load_vgg16('models')
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

img = Image.open("frame_1.png")
trns = transforms.ToTensor()
img_vgg = vgg_preprocess(trns(img).unsqueeze(0))
img_fea = vgg(img_vgg)
print(img_fea.shape)