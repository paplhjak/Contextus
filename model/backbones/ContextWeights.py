import torch
import torch.nn as nn
from model.backbones.functional import laplacian_context_mask, gaussian_context_mask, unfolded_context, inv_l2_context_mask


class GaussianContextWeights(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, dilation=1, initial_sigma=50.0, learnable_sigma=True):
        super(GaussianContextWeights, self).__init__()

        # kernel_size
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "expected kernel_size to contain 2 elements"
            assert kernel_size[0] == kernel_size[1], "expected kernel_size to be of size N x N"
            self.kernel_size = kernel_size[0]
        else:
            assert isinstance(kernel_size, int), "expected kernel_size to be of type int"
            self.kernel_size = kernel_size

        # stride
        if isinstance(stride, tuple):
            assert len(stride) == 2, "expected stride to contain 2 elements"
            assert stride[0] == stride[1], "expected stride to be of size N x N"
            self.stride = stride[0]
        else:
            assert isinstance(stride, int), "expected stride to be of type int"
            self.stride = stride

        # padding
        if isinstance(padding, tuple):
            assert len(padding) == 2, "expected padding to contain 2 elements"
            assert padding[0] == padding[1], "expected padding to be of size N x N"
            self.padding = padding[0]
        else:
            assert isinstance(padding, int), "expected padding to be of type int"
            self.padding = padding

        # dilation
        if isinstance(dilation, tuple):
            assert len(dilation) == 2, "expected dilation to contain 2 elements"
            assert dilation[0] == dilation[1], "expected dilation to be of size N x N"
            self.dilation = dilation[0]
        else:
            assert isinstance(dilation, int), "expected dilation to be of type int"
            self.dilation = dilation

        # context weighting param
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(initial_sigma), requires_grad=True)
            self.register_parameter('sigma',self.sigma)
        else:
            self.sigma = initial_sigma

    def forward(self, input):
        unfolded, center = unfolded_context(input, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation)
        mask = gaussian_context_mask(unfolded, center, sigma=self.sigma)
        return mask

class LaplacianContextWeights(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, dilation=1, initial_sigma=50.0, learnable_sigma=True):
        super(LaplacianContextWeights, self).__init__()

        # kernel_size
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "expected kernel_size to contain 2 elements"
            assert kernel_size[0] == kernel_size[1], "expected kernel_size to be of size N x N"
            self.kernel_size = kernel_size[0]
        else:
            assert isinstance(kernel_size, int), "expected kernel_size to be of type int"
            self.kernel_size = kernel_size

        # stride
        if isinstance(stride, tuple):
            assert len(stride) == 2, "expected stride to contain 2 elements"
            assert stride[0] == stride[1], "expected stride to be of size N x N"
            self.stride = stride[0]
        else:
            assert isinstance(stride, int), "expected stride to be of type int"
            self.stride = stride

        # padding
        if isinstance(padding, tuple):
            assert len(padding) == 2, "expected padding to contain 2 elements"
            assert padding[0] == padding[1], "expected padding to be of size N x N"
            self.padding = padding[0]
        else:
            assert isinstance(padding, int), "expected padding to be of type int"
            self.padding = padding

        # dilation
        if isinstance(dilation, tuple):
            assert len(dilation) == 2, "expected dilation to contain 2 elements"
            assert dilation[0] == dilation[1], "expected dilation to be of size N x N"
            self.dilation = dilation[0]
        else:
            assert isinstance(dilation, int), "expected dilation to be of type int"
            self.dilation = dilation

        # context weighting param
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(initial_sigma))
            self.register_parameter('sigma', self.sigma)
        else:
            self.sigma = initial_sigma


    def forward(self, input):
        unfolded, center = unfolded_context(input, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation)
        mask = laplacian_context_mask(unfolded, center, sigma=self.sigma)
        return mask

class InvL2ContextWeights(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, dilation=1, initial_a=0.01, learnable_a=True,
                 initial_b=1.0, learnable_b=False):
        super(InvL2ContextWeights, self).__init__()

        # kernel_size
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "expected kernel_size to contain 2 elements"
            assert kernel_size[0] == kernel_size[1], "expected kernel_size to be of size N x N"
            self.kernel_size = kernel_size[0]
        else:
            assert isinstance(kernel_size, int), "expected kernel_size to be of type int"
            self.kernel_size = kernel_size

        # stride
        if isinstance(stride, tuple):
            assert len(stride) == 2, "expected stride to contain 2 elements"
            assert stride[0] == stride[1], "expected stride to be of size N x N"
            self.stride = stride[0]
        else:
            assert isinstance(stride, int), "expected stride to be of type int"
            self.stride = stride

        # padding
        if isinstance(padding, tuple):
            assert len(padding) == 2, "expected padding to contain 2 elements"
            assert padding[0] == padding[1], "expected padding to be of size N x N"
            self.padding = padding[0]
        else:
            assert isinstance(padding, int), "expected padding to be of type int"
            self.padding = padding

        # dilation
        if isinstance(dilation, tuple):
            assert len(dilation) == 2, "expected dilation to contain 2 elements"
            assert dilation[0] == dilation[1], "expected dilation to be of size N x N"
            self.dilation = dilation[0]
        else:
            assert isinstance(dilation, int), "expected dilation to be of type int"
            self.dilation = dilation

        # context weighting param
        if learnable_a:
            self.a = nn.Parameter(torch.tensor(initial_a))
            self.register_parameter('a', self.a)
        else:
            self.a = initial_a
        if learnable_b:
            self.b = nn.Parameter(torch.tensor(initial_b))
            self.register_parameter('b', self.b)
        else:
            self.b = initial_b

    def forward(self, input):
        unfolded, center = unfolded_context(input, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation)
        mask = inv_l2_context_mask(unfolded, center, a=self.a, b=self.b)
        return mask
