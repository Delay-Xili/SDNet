'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dictnet import DictBlock
from Lib.config import config as _cfg

cfg = _cfg

class DictConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DictConv2d, self).__init__()

        self.dn = DictBlock(
            in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding,
            mu=cfg['MODEL']['MU'], lmbd=cfg['MODEL']['LAMBDA'][0], square_noise=cfg['MODEL']['SQUARE_NOISE'],
            n_dict=cfg['MODEL']['EXPANSION_FACTOR'], non_negative=cfg['MODEL']['NONEGATIVE'],
            n_steps=cfg['MODEL']['NUM_LAYERS'], FISTA=cfg['MODEL']['ISFISTA'], w_norm=cfg['MODEL']['WNORM']
        )

        # print(cfg)
        # print("==================")
        self.register_buffer('running_c_loss', torch.tensor(0, dtype=torch.float))

    def forward(self, x):
        out, rc = self.dn(x)
        if self.training:
            self.running_c_loss = 0.99 * self.running_c_loss + (1 - 0.99) * rc[1].item()

        return out

        # out, rc = self.dn(x)

        # delta_closs = rc[1] - self.running_c_loss
        # while delta_closs > 0.:
        #     print("in the loop")
        #     print(rc[1], self.running_c_loss)
        #     self.dn.lmbd += 0.1
        #     out, rc = self.dn(x)
        #     delta_closs = rc[1] - self.running_c_loss
        # return out

    def robust_forward(self, x):
        out, rc = self.dn(x)

        delta_closs = rc[1] - self.running_c_loss
        while delta_closs > 0.:
            print("in the loop")
            self.dn.lmbd += 0.1
            out, rc = self.dn(x)
            delta_closs = rc[1] - self.running_c_loss
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = DictConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DictConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                DictConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_conv(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_conv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = DictConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DictConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DictConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                DictConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # out = F.relu(out)
        out = self.bn2(self.conv2(out))
        # out = F.relu(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, Dataname=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if 'cifar' in Dataname:

            self.layer0 = nn.Sequential(
                DictConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
            )
        elif 'imagenet' in Dataname:
            self.layer0 = nn.Sequential(
                DictConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            raise ValueError()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

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


def SDNet18(num_classes, cfg):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, cfg["DATASET"]["DATASET"])


def SDNet18_sdlayer_only_for_first_layer(num_classes, cfg):
    return ResNet(BasicBlock_conv, [2, 2, 2, 2], num_classes, cfg["DATASET"]["DATASET"])


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