'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .dictnet import DictBlock
from .dictnet import DictBlock_pad as DictBlock
from Lib.config import config as _cfg

cfg = _cfg


class DictConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DictConv2d, self).__init__()
        assert stride <= 2
        self.in_stride = stride
        self.stride = 1 if cfg['MODEL']['POOLING'] else stride

        self.dn = DictBlock(
            in_channels, out_channels, stride=self.stride, kernel_size=kernel_size, padding=padding,
            mu=cfg['MODEL']['MU'], lmbd=cfg['MODEL']['LAMBDA'][0],
            n_dict=cfg['MODEL']['EXPANSION_FACTOR'], non_negative=cfg['MODEL']['NONEGATIVE'],
            n_steps=cfg['MODEL']['NUM_LAYERS'], FISTA=cfg['MODEL']['ISFISTA'], w_norm=cfg['MODEL']['WNORM'],
            padding_mode=cfg['MODEL']['PAD_MODE']
        )

    def forward(self, x):
        out, rc = self.dn(x)

        if cfg['MODEL']['POOLING'] and self.in_stride == 2:
            out = F.max_pool2d(out, kernel_size=2, stride=2)
        # if self.stride == 2:

        return out

    def inverse(self, z):
        z_tilde = z
        with torch.no_grad():

            if cfg['MODEL']['POOLING'] and self.in_stride == 2:
                z_tilde = F.interpolate(z_tilde, scale_factor=2, mode="bilinear")

            x_title = F.conv_transpose2d(
                z_tilde, self.dn.weight,
                bias=None, stride=self.dn.stride, padding=self.dn.padding,
                output_padding=self.dn.conv_transpose_output_padding
            )

        return x_title


class BatchNorm(nn.BatchNorm2d):

    def inverse(self, z):

        with torch.no_grad():
            bias = self.bias[None, :, None, None]
            weight = self.weight[None, :, None, None]
            if self.training:
                # for debug
                mean = self.running_mean[None, :, None, None]
                var = self.running_var[None, :, None, None]
                # raise ValueError()

            else:
                mean = self.running_mean[None, :, None, None]
                var = self.running_var[None, :, None, None]

            mid = (z - bias) / weight
            x_title = mid * (var ** 0.5) + mean

        return x_title


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = DictConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = DictConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)

        if cfg['MODEL']['SHORTCUT']:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        if cfg['MODEL']['SHORTCUT']:
            out += self.shortcut(x)
            out = F.relu(out)
        return out

    def inverse(self, z):
        if cfg['MODEL']['SHORTCUT']:
            raise ValueError()

        with torch.no_grad():
            x_title = self.bn2.inverse(z)
            x_title = self.conv2.inverse(x_title)
            x_title = self.bn1.inverse(x_title)
            x_title = self.conv1.inverse(x_title)
        return x_title


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = DictConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = DictConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = DictConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(self.expansion*planes)

        if cfg['MODEL']['SHORTCUT']:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if cfg['MODEL']['SHORTCUT']:
            out += self.shortcut(x)
            out = F.relu(out)
        return out

    def inverse(self, z):
        if cfg['MODEL']['SHORTCUT']:
            raise ValueError()

        with torch.no_grad():
            x_title = self.bn3.inverse(z)
            x_title = self.conv3.inverse(x_title)
            x_title = self.bn2.inverse(x_title)
            x_title = self.conv2.inverse(x_title)
            x_title = self.bn1.inverse(x_title)
            x_title = self.conv1.inverse(x_title)
        return x_title


class _make_layer(nn.Module):

    def __init__(self, block, in_planes, planes, num_blocks, stride):
        super(_make_layer, self).__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        self.layers = nn.Sequential(*layers)

    def forward(self, x, layer_th=None):

        if layer_th is not None:
            assert 1 <= layer_th <= len(self.layers)
            lid = [k for k in range(layer_th)]
        else:

            lid = [k for k in range(len(self.layers))]

        out = x
        for i in lid:
            out = self.layers[i](out)

        return out

    def inverse(self, z, layer_th=None):
        if layer_th is not None:
            assert 1 <= layer_th <= len(self.layers)
            lid = [k for k in range(layer_th)]
        else:

            lid = [k for k in range(len(self.layers))]

        x_title = z
        for i in reversed(lid):
            x_title = self.layers[i].inverse(x_title)

        return x_title


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, Dataname=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if 'cifar' in Dataname:

            self.layer0 = nn.Sequential(
                DictConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm(64),
                # nn.ReLU(inplace=True),
            )
        elif 'imagenet' in Dataname:
            self.layer0 = nn.Sequential(
                DictConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                BatchNorm(64),
                # nn.ReLU(inplace=True),
                DictConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError()
        self.num_blocks = num_blocks

        self.layer1 = _make_layer(block, self.in_planes, 64, num_blocks[0], stride=1)
        self.in_planes = 64 * block.expansion
        self.layer2 = _make_layer(block, self.in_planes, 128, num_blocks[1], stride=2)
        self.in_planes = 128 * block.expansion
        self.layer3 = _make_layer(block, self.in_planes, 256, num_blocks[2], stride=2)
        self.in_planes = 256 * block.expansion
        self.layer4 = _make_layer(block, self.in_planes, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def update_stepsize(self):

        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        # out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, None

    def _forward(self, x, layer_th):

        assert len(layer_th) == 2
        stage, block_th = layer_th
        assert 0 <= stage <= 4
        if stage >= 1:
            assert 1 <= block_th <= self.num_blocks[stage-1]

        out = self.layer0(x)
        if stage <= 0:
            return out

        for i in range(1, stage+1):

            if i == stage:
                out = eval(f"self.layer{i}(out, layer_th=block_th)")
            else:
                out = eval(f"self.layer{i}(out)")

        return out

    def _inverse(self, z, layer_th):

        assert len(layer_th) == 2
        stage, block_th = layer_th
        assert 0 <= stage <= 4
        if stage >= 1:
            assert 1 <= block_th <= self.num_blocks[stage - 1]

        if stage <= 0:
            x = z
            for k in range(len(self.layer0)-1, -1, -1):
                x = self.layer0[k].inverse(x)
            return x

        else:
            x = z
            for i in range(stage, 0, -1):

                if i == stage:
                    x = eval(f"self.layer{i}.inverse(x, layer_th=block_th)")
                else:
                    x = eval(f"self.layer{i}.inverse(x)")

            for k in range(len(self.layer0)-1, -1, -1):
                x = self.layer0[k].inverse(x)

            return x

    def generate_x(self, x, layer_th):
        with torch.no_grad():
            z       = self._forward(x, layer_th)
            x_title = self._inverse(z, layer_th)

            if cfg['VIZ_INPUTNORM']:
                if 'cifar' in cfg['DATASET']['DATASET']:
                    mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])[None, :, None, None].to('cuda')
                    std = torch.FloatTensor([0.2023, 0.1994, 0.2010])[None, :, None, None].to('cuda')
                elif 'imagenet' in cfg['DATASET']['DATASET']:
                    mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
                    std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
                else:
                    raise ValueError()

                x_title = x_title * std + mean

            # normalize the image energy  NCHW
            x_max = x_title.amax(dim=2, keepdim=True).amax(dim=3, keepdim=True)
            x_min = x_title.amin(dim=2, keepdim=True).amin(dim=3, keepdim=True)
            x_norm = (x_title - x_min) / (x_max - x_min)

            x_hist_norm = batch_hist_equalize(x_norm.cpu()).cuda()

        return z, x_title, x_norm, x_hist_norm


def batch_hist_equalize(images):
    # input requied image: [N, C, H, W] and scale in [0,1]
    #

    # 1 change NCHW -> NHWC
    n, c, h, w = images.size()
    x = images.permute(0, 2, 3, 1) * 255
    xx = [torch_equalize(x[i])[None] for i in range(n)]
    histed_image = torch.cat(xx, 0) / 255.

    return histed_image.permute(0, 3, 1, 2)


def torch_equalize(image):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352"""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)

        return result.type(torch.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = torch.stack([s1, s2, s3], 2)
    return image


def SDNet18(num_classes, cfg):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, cfg["DATASET"]["DATASET"])


def SDNet34(num_classes, cfg):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, cfg["DATASET"]["DATASET"])


def SDNet50(num_classes, cfg):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, cfg["DATASET"]["DATASET"])


def SDNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def SDNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':

    cfg = {'DATASET': {'DATASET': 'imagenet'}}
    net = SDNet50(num_classes=100, cfg=cfg)
    y = net(torch.randn(1, 3, 224, 224))

    params_sum = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f"params_sum: {params_sum / 1000000.0} M")

    print(y.size())