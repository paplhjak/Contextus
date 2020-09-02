from __future__ import absolute_import, division, print_function
from enum import Enum
import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn.functional as fn
import unittest

__all__ = [
    'conv_size',
    'conv_shape',
    'full_mask',
    'is_seq',
    'MaskedConvScaling',
    'masked_conv2d'
]


def is_seq(obj):
    """Is a sequence addressable via [] operator."""
    return hasattr(obj, "__getitem__")


def conv_size(size, kernel_size, stride=1, padding=0, dilation=1):
    """Compute output size of conv* (single dimension)."""
    return (size + (2 * padding) - dilation * (kernel_size - 1) - 1) // stride + 1


def conv_shape(shape, kernel_size, stride=1, padding=0, dilation=1):
    """Compute output shape of conv* (spatial dimensions)."""
    ndim = len(shape)
    if not is_seq(kernel_size):
        kernel_size = ndim * (kernel_size,)
    if not is_seq(stride):
        stride = ndim * (stride,)
    if not is_seq(padding):
        padding = ndim * (padding,)
    if not is_seq(dilation):
        dilation = ndim * (dilation,)
    output_shape = []
    for n, k, s, p, d in zip(shape, kernel_size, stride, padding, dilation):
        output_shape.append(conv_size(n, k, s, p, d))
    return type(shape)(output_shape)


#
# Masked convolution
#

class MaskedConvScaling(Enum):
    """Mode of scaling masked convolution output."""
    none = 'none'
    unit = 'unit'
    inputs = 'inputs'


def full_mask(shape, kernel_size, stride=1, padding=0, dilation=1):
    """Create full mask for masked convolution. See masked_conv2d."""
    n, c = shape[:2]
    ndim = len(shape) - 2
    if not is_seq(kernel_size):
        kernel_size = ndim * (kernel_size,)
    shape = (n,) + kernel_size + tuple(conv_shape(shape[2:], kernel_size, stride, padding, dilation))
    mask = torch.ones(shape, dtype=torch.float32)
    return mask


def unfolded_context(input, kernel_size, stride=1, padding=0, dilation=1):
    """Unfolded context for mask computation (only 2D due to unfold)."""
    # input (n, ic, ih, iw)
    # kernel_size (kh, kw)
    ndim = len(input.shape) - 2
    if not is_seq(kernel_size):
        kernel_size = ndim * (kernel_size,)
    c_sub = (np.array(kernel_size) - 1) // 2
    n, ic = input.shape[:2]
    input_shape = input.shape[2:]
    output_shape = conv_shape(input_shape, kernel_size, stride=stride, padding=padding, dilation=dilation)
    input = fn.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    # (n, ic*k, o)
    input = input.reshape((n, ic) + kernel_size + output_shape)
    # (n, ic, k0, k1, ..., o0, o1, ...)
    # TODO: Generic indexing for any dimensions.
    if ndim == 1:
        center = input[:, :, c_sub[0]]
    elif ndim == 2:
        center = input[:, :, c_sub[0], c_sub[1]]
    elif ndim == 3:
        center = input[:, :, c_sub[0], c_sub[1], c_sub[2]]
    center = center.reshape((n, ic) + ndim * (1,) + output_shape)
    return input, center


def gaussian_context_mask(input, center, sigma, normalize=False):
    """Gaussian context mask, exp(1/2* [(input-center)/sigma)]^2."""
    # input  (n, ic, k0, k1, ..., o0, o1, ...)
    # center (n, ic,  1,  1, ..., o0, o2, ...)
    ic = input.shape[1]
    # Allow shared sigma for all channels (-1).
    ic_shape = (1, -1) + (len(input.shape) - 2) * (1,)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)
    elif not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma)
    sigma = sigma.reshape(ic_shape).cuda()
    mask = torch.exp(-0.5 * (((input - center) / sigma)**2).sum(dim=1))
    if normalize:
        mask = mask / mask.sum(dim=(1, 2), keepdim=True)
    return mask

def laplacian_context_mask(input, center, sigma, normalize=False):
    """Laplacian context mask, 1/(2b)*exp(-abs(input-center)/sigma))."""
    # input  (n, ic, k0, k1, ..., o0, o1, ...)
    # center (n, ic,  1,  1, ..., o0, o2, ...)
    ic = input.shape[1]
    # Allow shared sigma for all channels (-1).
    ic_shape = (1, -1) + (len(input.shape) - 2) * (1,)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)
    elif not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma)
    sigma = sigma.reshape(ic_shape).cuda()
    mask = torch.exp(((-abs(input - center) / sigma)).sum(dim=1))
    if normalize:
        mask = mask / mask.sum(dim=(1, 2), keepdim=True)
    return mask


def inv_l2_context_mask(input, center, a=1.0, b=1.0, normalize=False):
    """Inverse L2 context mask, 1 / (a * l2(input, center) + b)."""
    # input  (n, ic, k0, k1, ..., o0, o1, ...)
    # center (n, ic,  1,  1, ..., o0, o2, ...)
    mask = (input - center).norm(dim=1)
    mask = 1. / (a * mask + b)
    if normalize:
        mask = mask / mask.sum(dim=(1, 2), keepdim=True)
    return mask


def masked_conv2d(input, weight, mask,
                  bias=None, stride=1, padding=0, dilation=1, groups=1,
                  scaling=MaskedConvScaling.none, eps=1e-6):
    """Masked 2D convolution using point-wise local context masks.

    Notation:
    n   batch size
    ic  input channels
    ih  image height
    iw  image width
    oc  output channels
    oh  output height
    ow  output width
    o   output elements (oh * ow)
    g   groups
    kw  kernel height
    kw  kernel width
    k   kernel elements (kh * kw)

    :param input:   Input image of shape (n, ic, ih, iw).
    :param weight:  Convolution kernel weight of shape (oc, ic/g, kh, kw).
    :param mask:    Local context masks of shape (n, kh, kw, oh, ow)
                    compatible with unfolded input.
    :param bias:    Has additive bias term?
    :param stride:  Step stride?
    :param padding: Input padding extent.
    :param dilation:    Kernel dilation?
    :param groups:  Number of input channel groups.
    :param scaling: Masked convolution scaling (none, unit, inputs).
    :param eps:     Epsilon to avoid division by zero.
    :return:        Masked convolution of shape (n, oc, oh, ow).
    """

    #if input.ndim != 4:
    #    raise ValueError('Only 2D masked convolution supported. '
    #                     'Input shape must be (n, c, h, w).')

    n, ic, ih, iw = input.shape
    oc, ic_g, kh, kw = weight.shape
    k = kh * kw
    if bias is not None:
        if bias.numel() not in (1, oc):
            raise ValueError('Number of bias elements (%i) '
                             'inconsistent with kernel output channels (%i).'
                             % (bias.numel(), oc))
    g = ic // ic_g

    assert ic % g == 0
    assert oc % g == 0
    if g != groups:
        raise ValueError('Number of groups in kernel (%i) '
                         'inconsistent with parameter (%i).'
                         % (g, groups))
    if n != mask.shape[0]:
        raise ValueError('Mask mini batch (%i)'
                         'inconsistent with that of input (%i).'
                         % (mask.shape[0], n))
    if (kh, kw) != tuple(mask.shape[1:3]):
        raise ValueError('Mask kernel size (%i, %i) '
                         'inconsistent with that of weight (%i, %i).'
                         % (kh, kw, mask.shape[1], mask.shape[2]))
    oh, ow = mask.shape[-2:]

    input = fn.unfold(input, (kh, kw),
                      dilation=dilation, padding=padding, stride=stride)
    # Input shape (n, ic * kh * kw, o), where o is number of output locations
    # considering input size, stride, padding, and dilation.
    o = input.shape[-1]
    if oh * ow != o:
        raise ValueError('Mask spatial size (%i) '
                         'inconsistent with unfolded input size (%i).'
                         % (oh * ow, o))
    assert o == oh * ow

    # Reshape for element-wise conv1d.
    # Reduce memory footprint: avoid full shape expansion during broadcasting.
    # Dimensions:          0,  1, 2, 3
    input = input.reshape((n, ic, k, o))
    mask  = mask .reshape((n,  1, k, o))
    masked_input = (input * mask).reshape((n, ic * k, o))
    weight = weight.reshape((oc, ic_g * k, 1))
    #masked_input.to(weight.device)
    output = fn.conv1d(masked_input, weight,
                       bias=None, stride=1, padding=0, dilation=1, groups=groups)
    assert (n, oc, o) == tuple(output.shape)

    # Scale output according to mask and scale.
    # NB: Mask is only single channel so channel dimension is not considered.
    if scaling != MaskedConvScaling.none:
        # scaling in (MaskedConvScaling.inputs, MaskedConvScaling.unit)
        mask_sum = mask.reshape((n, k, o)).sum(dim=1, keepdim=True)
        output = output / (mask_sum + eps)
    if scaling == MaskedConvScaling.inputs:
        output = k * output

    # Add bias term if provided (allow shared scalar bias).
    if bias is not None:
        output = output + bias.reshape((1, -1, 1))

    output = output.reshape((n, oc, oh, ow))

    return output


#
# Unit tests
#

class TestIsSeq(unittest.TestCase):

    # def is_seq(obj):
    #     """Is a sequence addressable via [] operator."""

    def test_str(self):
        self.assertTrue(is_seq(""))
        self.assertTrue(is_seq("abc"))

    def test_tuple(self):
        self.assertTrue(is_seq(()))
        self.assertTrue(is_seq((1, 2, 3)))

    def test_list(self):
        self.assertTrue(is_seq([]))
        self.assertTrue(is_seq([1, 2, 3]))

    def test_int(self):
        self.assertFalse(is_seq(1))
        self.assertFalse(is_seq(-1))


class TestConvSize(unittest.TestCase):

    # def conv_size(size, kernel_size, stride=1, padding=0, dilation=1):
    #     """Compute output size of conv* (single dimension)."""

    def test_k1_defaults(self):
        self.assertEqual(conv_size(5, 1), 5)

    def test_k1_pad1(self):
        self.assertEqual(conv_size(5, 1, padding=1), 7)

    def test_k3_defaults(self):
        self.assertEqual(conv_size(5, 3), 3)

    def test_k3_pad1(self):
        self.assertEqual(conv_size(5, 3, padding=1), 5)


class TestConvShape(unittest.TestCase):

    # def conv_shape(shape, kernel_size, stride=1, padding=0, dilation=1):
    #     """Compute output shape of conv* (spatial dimensions)."""

    def test_1d_k1_defaults(self):
        self.assertEqual(conv_shape((5,), 1), (5,))
        self.assertEqual(conv_shape((5,), (1,)), (5,))

    def test_1d_k1_pad1(self):
        self.assertEqual(conv_shape((5,), 1, padding=1), (7,))
        self.assertEqual(conv_shape((5,), (1,), padding=(1,)), (7,))

    def test_2d_k1_defaults(self):
        self.assertEqual(conv_shape((5, 5), 1), (5, 5))
        self.assertEqual(conv_shape((5, 5), (1, 1)), (5, 5))

    def test_2d_k1_pad1(self):
        self.assertEqual(conv_shape((5, 5), 1, padding=1), (7, 7))
        self.assertEqual(conv_shape((5, 5), (1, 1), padding=(1, 1)), (7, 7))

    def test_2d_k3_defaults(self):
        self.assertEqual(conv_shape((5, 5), 3), (3, 3))
        self.assertEqual(conv_shape((5, 5), (3, 3)), (3, 3))

    def test_2d_k3_pad1(self):
        self.assertEqual(conv_shape((5, 5), 3, padding=1), (5, 5))
        self.assertEqual(conv_shape((5, 5), (3, 3), padding=(1, 1)), (5, 5))


class TestFullMask(unittest.TestCase):

    # def full_mask(shape, kernel_size, stride=1, padding=0, dilation=1, mode=MaskedConvMode.scaled):
    #     """Create full mask for masked convolution."""
    # mask size (n, kh, kw, oh, ow)

    def test_1d(self):
        n = 1
        c = 1
        self.assertEqual(full_mask((n, c, 5), 1).shape, (1, 1, 5))
        self.assertTrue((full_mask((n, c, 5), 1) == 1.).all())

        self.assertEqual(full_mask((n, c, 5), 1, padding=1).shape, (1, 1, 7))
        self.assertTrue((full_mask((n, c, 5), 1, padding=1) == 1.).all())

        self.assertEqual(full_mask((n, c, 5), 3).shape, (1, 3, 3))
        self.assertTrue((full_mask((n, c, 5), 3) == 1.).all())

        self.assertEqual(full_mask((n, c, 5), 3, padding=1).shape, (1, 3, 5))
        self.assertTrue((full_mask((n, c, 5), 3, padding=1) == 1.).all())

    def test_2d(self):
        n = 1
        c = 1
        self.assertEqual(full_mask((n, c, 5, 5), 3).shape, (1, 3, 3, 3, 3))
        self.assertTrue((full_mask((n, c, 5, 5), 3) == 1.).all())

        self.assertEqual(full_mask((n, c, 5, 5), 3, padding=1).shape, (1, 3, 3, 5, 5))
        self.assertTrue((full_mask((n, c, 5, 5), 3, padding=1) == 1.).all())


class TestUnfoldedContext(unittest.TestCase):

    def test_defaults(self):
        n = 1
        ic = 2
        kh, kw = 3, 3
        ih, iw = kh, 5
        x = torch.arange(n * ic * ih * iw, dtype=torch.float32).reshape((n, ic, ih, iw))
        y, c = unfolded_context(x, (kw, kw))
        self.assertEqual(c.flatten().tolist(), [6., 7., 8., 21., 22., 23.])

        kh, kw = 2, 2
        ih, iw = kh, 4
        x = torch.arange(n * ic * ih * iw, dtype=torch.float32).reshape((n, ic, ih, iw))
        y, c = unfolded_context(x, (kw, kw))
        self.assertEqual(c.flatten().tolist(), [0., 1., 2., 8., 9., 10.])


class TestGaussianContextMask(unittest.TestCase):

    def test_defaults(self):
        n = 1
        ic = 1
        kh, kw = 2, 2
        ih, iw = kh, kw + 1
        x = torch.arange(n * ic * ih * iw, dtype=torch.float32).reshape((n, ic, ih, iw))
        # n, ic, ih, iw
        y, c = unfolded_context(x, (kw, kw))
        # n, ic, kh, kw, oh, ow
        m = gaussian_context_mask(y, c, 1.0)
        # n, kh, kw, oh, ow
        self.assertEqual(m[0, 0, 0, 0, 0], 1.0)
        self.assertAlmostEqual(m[0, 0, 1, 0, 0], 0.606530660, 6)

        ic = 2
        kh, kw = 3, 3
        ih, iw = kh, kw + 1
        x = torch.arange(n * ic * ih * iw, dtype=torch.float32).reshape((n, ic, ih, iw))
        # n, ic, ih, iw
        y, c = unfolded_context(x, (kw, kw))
        # n, ic, kh, kw, oh, ow
        m = gaussian_context_mask(y, c, 1.0)
        # n, kh, kw, oh, ow
        self.assertEqual(m[0, 1, 1, 0, 0], 1.0)


class TestInvL2ContextMask(unittest.TestCase):

    def test_defaults(self):
        n = 1
        ic = 1
        kh, kw = 2, 2
        ih, iw = kh, kw + 1
        x = torch.arange(n * ic * ih * iw, dtype=torch.float32).reshape((n, ic, ih, iw))
        # n, ic, ih, iw
        y, c = unfolded_context(x, (kw, kw))
        # n, ic, kh, kw, oh, ow
        m = inv_l2_context_mask(y, c)
        # n, kh, kw, oh, ow
        self.assertEqual(m[0, 0, 0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(m[0, 0, 1, 0, 0].item(), 0.5, 6)
        self.assertAlmostEqual(m[0, 1, 0, 0, 0].item(), 0.25, 6)
        self.assertAlmostEqual(m[0, 1, 1, 0, 0].item(), 0.2, 6)

        ic = 2
        kh, kw = 3, 3
        ih, iw = kh, kw + 1
        x = torch.arange(n * ic * ih * iw, dtype=torch.float32).reshape((n, ic, ih, iw))
        # n, ic, ih, iw
        y, c = unfolded_context(x, (kw, kw))
        # n, ic, kh, kw, oh, ow
        m = inv_l2_context_mask(y, c)
        # n, kh, kw, oh, ow
        self.assertEqual(m[0, 1, 1, 0, 0], 1.0)


class TestMaskedConv2d(unittest.TestCase):

    # def masked_conv2d(input, weight, mask,
    #                   bias=None, stride=1, padding=0, dilation=1, groups=1,
    #                   mode=MaskedConvMode.scaled):
    #
    #     # input (n, ic, ih, iw)         input image
    #     # weight (oc, ic/g, kh, kw)     convolution kernel weight
    #     # # mask (n, ih, iw, kh, kw)      local attention
    #     # mask (n, kh, kw, oh, ow)      local attention, already output sized
    #     # n number of batches
    #     # ic input channels
    #     # ih image height
    #     # iw image width
    #     # oc output channels
    #     # g groups
    #     # kw kernel height
    #     # kw kernel width

    def test_input_support_check(self):
        n = 1
        ic = 1
        oc = 1
        kw = {}

        x = torch.rand((n, ic, 5))
        w = torch.rand((oc, ic, 3))
        m = full_mask(x.shape, w.shape[2:], **kw)
        self.assertRaises(ValueError, masked_conv2d, x, w, m, **kw)

        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic, 3, 3))
        m = full_mask(x.shape, w.shape[2:], **kw)
        self.assertEqual(masked_conv2d(x, w, m, **kw).shape[2:], conv_shape(x.shape[2:], w.shape[2:], **kw))

        x = torch.rand((n, ic, 5, 5, 5))
        w = torch.rand((oc, ic, 3, 3, 3))
        m = full_mask(x.shape, w.shape[2:], **kw)
        self.assertRaises(ValueError, masked_conv2d, x, w, m, **kw)

    def test_2d_output(self):
        n = 1
        ic = 1
        oc = 1
        places = 4

        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic, 3, 3))
        m = full_mask(x.shape, w.shape[2:])

        # conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        self.assertAlmostEqual((masked_conv2d(x, w, m) - fn.conv2d(x, w)).norm().item(), 0., places)

        ic = 1
        oc = 2
        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic, 3, 3))
        self.assertAlmostEqual((masked_conv2d(x, w, m) - fn.conv2d(x, w)).norm().item(), 0., places)

        ic = 2
        oc = 1
        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic, 3, 3))
        self.assertAlmostEqual((masked_conv2d(x, w, m) - fn.conv2d(x, w)).norm().item(), 0., places)

        ic = 2
        oc = 3
        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic, 3, 3))
        self.assertAlmostEqual((masked_conv2d(x, w, m) - fn.conv2d(x, w)).norm().item(), 0., places)

        g = 2
        ic = g
        oc = g
        kw = {'groups': g}
        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic // g, 3, 3))
        m = full_mask(x.shape, w.shape[2:])
        self.assertAlmostEqual((masked_conv2d(x, w, m, **kw) - fn.conv2d(x, w, **kw)).norm().item(), 0., places)

        g = 2
        ic = 2 * g
        oc = 1 * g
        kw = {'groups': g}
        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic // g, 3, 3))
        m = full_mask(x.shape, w.shape[2:])
        self.assertAlmostEqual((masked_conv2d(x, w, m, **kw) - fn.conv2d(x, w, **kw)).norm().item(), 0., places)

        g = 2
        ic = 1 * g
        oc = 3 * g
        kw = {'groups': g}
        x = torch.rand((n, ic, 5, 5))
        w = torch.rand((oc, ic // g, 3, 3))
        m = full_mask(x.shape, w.shape[2:])
        self.assertAlmostEqual((masked_conv2d(x, w, m, **kw) - fn.conv2d(x, w, **kw)).norm().item(), 0., places)

        g = 2
        ic = 2 * g
        oc = 3 * g
        kw = {'groups': g}
        x = torch.rand((n, ic, 5, 5))
        # x = torch.rand((n, ic, 320, 320))
        w = torch.rand((oc, ic // g, 3, 3))
        m = full_mask(x.shape, w.shape[2:])
        self.assertAlmostEqual((masked_conv2d(x, w, m, **kw)
                                - fn.conv2d(x, w, **kw)).norm().item(), 0., places)
        self.assertAlmostEqual((masked_conv2d(x, w, m, scaling=MaskedConvScaling.inputs, **kw)
                                - fn.conv2d(x, w, **kw)).norm().item(), 0., places)

        # Large input layer
        g = 16
        ic = 64
        oc = 2 * ic
        kw = {'groups': g}
        x = torch.rand((n, ic, 512, 512))
        w = torch.rand((oc, ic // g, 3, 3))
        m = full_mask(x.shape, w.shape[2:])
        t = timer()
        out_masked = masked_conv2d(x, w, m, **kw)
        t = timer() - t
        # print('Runtime: %.3f s' % t)
        self.assertLessEqual(t, 2.)
        out_conv2d = fn.conv2d(x, w, **kw)
        self.assertAlmostEqual((out_masked - out_conv2d).norm().item()
                               / min(out_masked.norm().item(), out_conv2d.norm().item()),
                               0., delta=1e-6)

        # Wide intermediate layer
        g = 2
        ic = 1024
        oc = 2 * ic
        kw = {'groups': g}
        x = torch.rand((n, ic, 64, 64))
        w = torch.rand((oc, ic // g, 3, 3))
        m = full_mask(x.shape, w.shape[2:])
        t = timer()
        out_masked = masked_conv2d(x, w, m, **kw)
        t = timer() - t
        # print('Runtime: %.3f s' % t)
        self.assertLessEqual(t, 2.)
        out_conv2d = fn.conv2d(x, w, **kw)
        self.assertAlmostEqual((out_masked - out_conv2d).norm().item()
                               / min(out_masked.norm().item(), out_conv2d.norm().item()),
                               0., delta=1e-6)


if __name__ == '__main__':
    unittest.main()