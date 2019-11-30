import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import EncoderAtt

encatt = EncoderAtt()
encatt.cuda()
inp = torch.ones((4,3,2)).cuda()
feat = encatt(inp)
print(feat.shape)