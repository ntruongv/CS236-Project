from utils import vgg_preprocess, load_vgg16, LocalGraph
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms

# vgg_model_path = ""

img = Image.open("frame_1.png")
lgph = LocalGraph(img)
# pass batch of (x,y) for local features
print(lgph.extract_batch([[25,25], [30, 30]]).shape)
