from __future__ import absolute_import, division, print_function
from model.backbones.functional import MaskedConvScaling
from model.backbones.functional import gaussian_context_mask
from model.backbones.functional import unfolded_context
import torch.nn.functional as fn
import torch

n, ic, ih, iw = 1, 1, 5, 5
oc, kh, kw = 1, 3, 3

bias = None
stride = 1
padding = 0
dilation = 1
groups = 1
scaling = MaskedConvScaling.none
eps = 1e-6
sigma = 1

input = torch.ones(n, ic, ih, iw).cuda()
weight = torch.ones(oc, int(ic / groups), kh, kw).cuda()
context_source = torch.randn(n, ic, ih, iw).cuda()
context_source[0,0,2,2] = 0
unfolded, center = unfolded_context(context_source, kernel_size=kh, stride=stride,
                                    padding=padding, dilation=dilation)

mask = gaussian_context_mask(unfolded, center, sigma=sigma)

k = kh * kw

oh, ow = mask.shape[-2:]

input = fn.unfold(input, (kh, kw),
                  dilation=dilation, padding=padding, stride=stride)
# Input shape (n, ic * kh * kw, o), where o is number of output locations
# considering input size, stride, padding, and dilation.
o = input.shape[-1]

# Reshape for element-wise conv1d.
# Reduce memory footprint: avoid full shape expansion during broadcasting.
# Dimensions:          0,  1, 2, 3
input = input.reshape((n, ic, k, o))
mask = mask.reshape((n, 1, k, o))
masked_input = (input * mask).reshape((n, ic * k, o))
weight = weight.reshape((oc, int(ic / groups) * k, 1))


# masked_input.to(weight.device)
output = fn.conv1d(masked_input, weight,
                   bias=None, stride=1, padding=0, dilation=1, groups=groups)
assert (n, oc, o) == tuple(output.shape)

# Scale output according to mask and scale.
# NB: Mask is only single channel so channel dimension is not considered.
if scaling != MaskedConvScaling.none:
    # scaling in (MaskedConvScaling.inputs, MaskedConvScaling.unit)
    mask_sum = mask.reshape((n, k, o)).sum(dim=1, keepdim=True)
    print(mask_sum)
    output = output / (mask_sum + eps)
    print(output)
if scaling == MaskedConvScaling.inputs:
    output = k * output

# Add bias term if provided (allow shared scalar bias).
if bias is not None:
    output = output + bias.reshape((1, -1, 1))

output = output.reshape((n, oc, oh, ow))
print(output)
