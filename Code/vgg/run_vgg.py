from utils import LocalGraph
from PIL import Image
import numpy as np

# vgg_model_path = ""

img = Image.open("frame_1.png")
h,w = img.size
lgph = LocalGraph(img, cell_h=24, cell_gap=16)
# pass batch of (x,y) for local features
pts = np.zeros((100,2))
pts[:,1] = np.random.randint(h,size=100)
pts[:,0] = np.random.randint(w,size=100)
print(pts.shape)
print(lgph.extract_batch(pts).shape)