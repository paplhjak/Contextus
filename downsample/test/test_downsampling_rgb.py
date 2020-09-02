from ..center import Net as cNet
from ..average import Net as aNet
from ..bilinear import Net as bNet
from ..nearest import Net as nNet
from PIL import Image
import torch
import numpy as np
import os

rgb_path = os.path.dirname(__file__)+"/test_image_in/rgb.png"
rgb_pil = Image.open(rgb_path)
rgb = np.array(rgb_pil, dtype=np.int)[:, :, np.newaxis]
rgb_new = rgb[:,:,0,:]
rgb_new = torch.from_numpy(rgb_new).permute(2,0,1).float().cuda()

"""
downsample center
"""
cnet = cNet(3)
outputs = cnet(rgb_new.unsqueeze(0))

for idx in range(len(outputs)):
    tmp = np.asarray(outputs[idx][0].permute((1,2,0)).cpu())
    tmp = (np.clip(tmp, 0, 255)).astype('u1')
    im = Image.fromarray(tmp)
    im.save(os.path.dirname(__file__)+'/test_image_out/' + str(idx) + '_center_rgb.png')

"""
downsample average
"""
anet = aNet(3)
outputs = anet(rgb_new.unsqueeze(0))

for idx in range(len(outputs)):
    tmp = np.asarray(outputs[idx][0].permute((1,2,0)).cpu())
    tmp = (np.clip(tmp, 0, 255)).astype('u1')
    im = Image.fromarray(tmp)
    im.save(os.path.dirname(__file__)+'/test_image_out/' + str(idx) + '_average_rgb.png')


"""
downsample bilinear
"""
bnet = bNet()
outputs = bnet(rgb_new.unsqueeze(0))

for idx in range(len(outputs)):
    tmp = np.asarray(outputs[idx][0].permute((1,2,0)).cpu())
    tmp = (np.clip(tmp, 0, 255)).astype('u1')
    im = Image.fromarray(tmp)
    im.save(os.path.dirname(__file__)+'/test_image_out/' + str(idx) + '_bilinear_rgb.png')


"""
downsample nearest
"""
nnet = nNet()
outputs = nnet(rgb_new.unsqueeze(0))

for idx in range(len(outputs)):
    tmp = np.asarray(outputs[idx][0].permute((1,2,0)).cpu())
    tmp = (np.clip(tmp, 0, 255)).astype('u1')
    im = Image.fromarray(tmp)
    im.save(os.path.dirname(__file__)+'/test_image_out/' + str(idx) + '_nearest_rgb.png')