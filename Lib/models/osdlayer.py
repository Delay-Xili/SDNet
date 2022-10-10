from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import torch.nn.init as init
from Lib.config import config


class elasnet_prox(nn.Module):
    r"""Applies the elastic net proximal operator,
    NOTS: it will degenerate to ell1_prox if mu=0.0

    The elastic net proximal operator function is given as the following function
    \argmin_{x} \lambda ||x||_1 + \mu /2 ||x||_2^2 + 0.5 ||x - input||_2^2

    Args:
      lambd: the :math:`\lambda` value on the ell_1 penalty term. Default: 0.5
      mu:    the :math:`\mu` value on the ell_2 penalty term. Default: 0.0

    Shape:
      - Input: :math:`(N, *)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(N, *)`, same shape as the input

    """

    def __init__(self, lambd=0.5, mu=0.0):
        super(elasnet_prox, self).__init__()
        self.lambd = lambd
        self.scaling_mu = 1.0 / (1.0 + mu)

    def forward(self, input):
        return F.softshrink(input * self.scaling_mu, self.lambd * self.scaling_mu)

    def extra_repr(self):
        return '{} {}'.format(self.lambd, self.scaling_mu)


class DictBlock(nn.Module):
    # c = argmin_c lmbd * ||c||_1  +  mu/2 * ||c||_2^2 + 1 / 2 * ||x - weight (@conv) c||_2^2
    def __init__(self, n_channel, dict_size, mu=0.0, lmbd=0.0, n_dict=1, non_negative=True,  # model parameters
                 stride=1, kernel_size=3, padding=1, share_weight=True,  # optional model parameters
                 n_steps=10, step_size_fixed=True, FISTA=True, step_size=0.1, w_norm=True, padding_mode="constant"):  # training parameters
        super(DictBlock, self).__init__()

        self.mu = mu
        self.lmbd = lmbd  # LAMBDA
        self.n_dict = n_dict
        self.stride = stride
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = padding
        self.padding_mode = padding_mode
        assert self.padding_mode in ['constant', 'reflect', 'replicate', 'circular']
        self.groups = 1
        self.n_steps = n_steps
        self.share_weight = share_weight
        self.FISTA = FISTA
        self.conv_transpose_output_padding = 0 if stride == 1 else 1
        self.w_norm = w_norm
        self.non_negative = non_negative

        self.xsize = None
        self.zsize = None
        self.lmbd_ = None

        n_variables = 1 if share_weight else self.n_steps
        self.weight = nn.Parameter(torch.Tensor(dict_size, self.n_dict * n_channel, kernel_size, kernel_size))

        with torch.no_grad():
            init.kaiming_uniform_(self.weight)

        # variables that are needed for ISTA/FISTA
        self.nonlinear = elasnet_prox(self.lmbd * step_size, self.mu * step_size)

        if step_size_fixed:
            self.step_size = [step_size for _ in range(n_variables)]
        else:
            self.step_size = nn.ParameterList(
                [nn.Parameter(torch.Tensor([step_size]))  # [math.sqrt(dict_size / n_channel)]))
                 for _ in range(n_variables)])

    def algorithm(self, x):

        for i in range(self.n_steps):
            index = 0 if self.share_weight else i
            weight = self.weight
            step_size = self.step_size[index]

            if i == 0:
                c_pre = 0.
                c = step_size * F.conv2d(x.repeat(1, self.n_dict, 1, 1), weight, bias=None, stride=self.stride,
                                         padding=self.padding)
                c = self.nonlinear(c)
            else:
                c_pre = c
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = torch.tanh(x.repeat(1, self.n_dict, 1, 1) - xp)
                insign = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                # c = c + step_size * torch.tanh(insign)
                c = c + step_size * insign
                c = self.nonlinear(c)

            if self.non_negative:
                c = F.relu(c)

            # self.c_error.append(torch.sum((c - c_pre) ** 2) / c.shape[0])
        return c, weight

    def forward(self, x):

        if self.w_norm:
            self.normalize_weight()

        c, weight = self.algorithm(x)

        # Compute loss
        xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                output_padding=self.conv_transpose_output_padding)
        r = x.repeat(1, self.n_dict, 1, 1) - xp
        r_loss = torch.sum(torch.abs(r)) / self.n_dict
        c_loss = self.lmbd * torch.sum(torch.abs(c)) + self.mu / 2. * torch.sum(torch.pow(c, 2))

        return c, (r_loss, c_loss)

    def normalize_weight(self):
        with torch.no_grad():
            w = self.weight.view(self.weight.size(0), -1)
            normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w)
            w = (w / normw).view(self.weight.size())
            self.weight.data = w.data
