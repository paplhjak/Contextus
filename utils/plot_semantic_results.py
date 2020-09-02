import numpy as np
from PIL import Image
from utils import cityscapes_labels

# Constants for drawing
BORDER = 10

COLORS_OK = np.array(((255, 0, 0, 0.3), (0, 255, 0, 0.3)))

def blend_img(background, overlay_rgba, gamma=0.2):  # taken from VIR
    alpha = overlay_rgba[:, :, 3]
    over_corr = np.float_power(overlay_rgba[:, :, :3], gamma)
    bg_corr = np.float_power(background, gamma)
    return np.float_power(over_corr * alpha[..., None] + (1 - alpha)[..., None] * bg_corr, 1 / gamma)  # dark magic
    # partially taken from https://en.wikipedia.org/wiki/Alpha_compositing#Composing_alpha_blending_with_gamma_correction


def create_vis(rgb, label, prediction):  # taken from VIR
    if rgb.shape[0] == 3:
        rgb = np.asarray(rgb).transpose(1, 2, 0)
     
    prediction = np.asarray(prediction)
    label = np.asarray(label)

    h, w, _ = rgb.shape
    #print(cityscapes_labels.label2rgba(label[0]).shape)
    gt_map = blend_img(rgb, cityscapes_labels.label2rgba(label))  # we can index colors, wohoo!
    pred_map = blend_img(rgb, cityscapes_labels.label2rgba(prediction))
        
    ok_or_not = COLORS_OK[
        (label == prediction).astype('u1')]
    ok_or_not[label==255]=(0,0,0, 0.3)
    
    ok_map = blend_img(rgb, ok_or_not)  # but we cannot do it by boolean, otherwise it won't work
            
    canvas = np.ones((h * 2 + BORDER, w * 2 + BORDER, 3))
    canvas[:h, :w] = rgb
    canvas[:h, -w:] = gt_map
    canvas[-h:, :w] = pred_map
    canvas[-h:, -w:] = ok_map

    canvas = (np.clip(canvas, 0, 255)).astype('u1')
    return Image.fromarray(canvas)
