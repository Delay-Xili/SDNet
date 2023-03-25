from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
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


class MatrixSparseCodingLayer(nn.Module):
    # c = argmin_c lmbd * ||c||_1  +  mu/2 * ||c||_2^2 + 1 / 2 * ||x - weight @ c||_2^2
    def __init__(self, n_channel, dict_size, mu=0.0, lmbd=0.1, n_dict=1, non_negative=True, n_steps=2, # model parameters
                 square_noise=True,  # optional model parameters
                 step_size=0.1, w_norm=True):  # training parameters
        super(MatrixSparseCodingLayer, self).__init__()

        self.mu = mu
        self.lmbd = lmbd  # LAMBDA
        self.n_dict = n_dict
        self.dict_size = dict_size
        self.n_channel = n_channel
        self.n_steps = n_steps
        self.w_norm = w_norm
        self.non_negative = non_negative
        self.v_max = None
        self.v_max_error = 0.
        # c = argmin_c lmbd * ||c||_1  +  mu/2 * ||c||_2^2 + 1 / 2 * ||x - weight @ c||_1 // if square noise is False
        self.square_noise = square_noise  #

        self.weight = nn.Parameter(torch.Tensor(dict_size, n_channel * self.n_dict))

        with torch.no_grad():
            init.kaiming_uniform_(self.weight)

        # variables that are needed for ISTA/FISTA
        self.nonlinear = elasnet_prox(self.lmbd * step_size, self.mu * step_size)

        self.register_buffer('step_size', torch.tensor(step_size, dtype=torch.float))

    def fista(self, x):

        for i in range(self.n_steps):

            weight = self.weight
            step_size = self.step_size

            if i == 0:
                c_pre = 0.
                # x [bs, n_channel * n_dict]; weight [dict_size, n_channel * n_dict]
                # where F.linear(x, A) execute the operator xA^T
                c = step_size * F.linear(x.repeat(1, self.n_dict), weight, bias=None)  # c [bs, dict_size]
                c = self.nonlinear(c)

            elif i == 1:
                c_pre = c
                # weight = self.normalize(weight)
                xp = F.linear(c, weight.T, bias=None)  # xp [bs, n_channel * n_dict]
                r = x.repeat(1, self.n_dict) - xp

                if self.square_noise:
                    gra = F.linear(r, weight, bias=None)
                else:

                    # w = r.view(r.size(0), -1)
                    # normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).detach()
                    # w = w / normw
                    r = r / r.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).detach()

                    gra = F.linear(r, weight) * 0.5

                c = c + step_size * gra
                c = self.nonlinear(c)
                t = (math.sqrt(5.0) + 1.0) / 2.0

            else:
                t_pre = t
                t = (math.sqrt(1.0 + 4.0 * t_pre * t_pre) + 1) / 2.0
                a = (t_pre + t - 1.0) / t * c + (1.0 - t_pre) / t * c_pre
                c_pre = c
                # weight = self.normalize(weight)
                xp = F.linear(c, weight.T, bias=None)
                r = x.repeat(1, self.n_dict) - xp

                if self.square_noise:
                    gra = F.linear(r, weight, bias=None)
                else:
                    r = r / r.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).detach()
                    gra = F.linear(r, weight) * 0.5

                c = a + step_size * gra
                c = self.nonlinear(c)

            if self.non_negative:
                c = F.relu(c)

            # self.c_error.append(torch.sum((c) ** 2) / c.shape[0])
        return c, weight

    def forward(self, x):

        '''
        :param x: [bs, token, dim] (for transformer) or [bs, dim] (for MLP)
        :return:
        '''

        if x.dim() == 2:
            bs, dim = x.shape
            x_ = x
        elif x.dim() == 3:
            bs, token, dim = x.shape
            x_ = x.view(-1, dim)
        else:
            raise NotImplementedError

        if self.training:
            self.update_stepsize()
            if torch.cuda.device_count() > 1:
                raise ValueError(
                    "would be cause conflict of number in Dataparallel!! "
                    "Move the update_stepsize before each feedforward of Dataparallel"
                )

        if self.w_norm and self.training:
            self.normalize_weight()

        c, weight = self.fista(x_)

        # Compute loss
        xp = F.linear(c, weight.T, bias=None)
        r = x_.repeat(1, self.n_dict) - xp
        r_loss = torch.sum(torch.pow(r, 2)) / self.n_dict
        c_loss = self.lmbd * torch.sum(torch.abs(c)) + self.mu / 2. * torch.sum(torch.pow(c, 2))

        if x.dim() == 2:
            pass
        elif x.dim() == 3:
            c = c.view(bs, token, -1)
            xp = xp.view(bs, token, -1)
        else:
            raise NotImplementedError

        return c, xp, r, (r_loss, c_loss)

    def update_stepsize(self):
        step_size = 0.9 / self.power_iteration(self.weight)
        self.step_size = self.step_size * 0. + step_size
        self.nonlinear.lambd = self.lmbd * step_size
        self.nonlinear.scaling_mu = 1.0 / (1.0 + self.mu * step_size)

    def normalize_weight(self):
        with torch.no_grad():
            w = self.weight.view(self.weight.size(0), -1)
            normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            w = w / normw
            self.weight.data = w.data

    def power_iteration(self, weight):

        max_iteration = 50
        v_max_error = 1.0e5
        tol = 1.0e-5
        k = 0

        with torch.no_grad():
            if self.v_max is None:
                v = torch.randn(size=(1, self.dict_size)).to(weight.device)
            else:
                v = self.v_max.clone()

            while k < max_iteration and v_max_error > tol:

                tmp = F.linear(v, weight.T, bias=None)
                v_ = F.linear(tmp, weight, bias=None)
                v_ = F.normalize(v_.view(-1), dim=0, p=2).view(v.size())
                v_max_error = torch.sum((v_ - v) ** 2)
                k += 1
                v = v_

            v_max = v.clone()
            Dv_max = F.linear(v_max, weight.T, bias=None)  # Dv
            lambda_max = torch.sum(Dv_max ** 2).item()  # vTDTDv / vTv, ignore the vTv since vTv = 1

        self.v_max = v_max
        return lambda_max


if __name__ == '__main__':

    # use case
    input_dim = 32
    output_dim = 64
    layer1 = MatrixSparseCodingLayer(n_channel=input_dim, dict_size=output_dim)

    x = torch.randn(1, 32)

    y = layer1(x)

    print(y)
