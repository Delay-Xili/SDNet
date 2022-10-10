import torch
import torch.nn as nn
import torch.nn.functional as F
from Lib.models.sdnet_inverse import DictConv2d, BatchNorm, batch_hist_equalize, DictBlock
from Lib.config import config as _cfg

cfg = _cfg


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        if cfg.MODEL.NAME == 'lenet4_viz':
            self.lenet4(num_classes)
            self.num_layer = 4
        elif cfg.MODEL.NAME == 'lenet8_viz':
            self.lenet8(num_classes)
            self.num_layer = 8
        else:
            raise ValueError()

    def lenet4(self, num_classes):
        self.layer1 = DictConv2d(in_channels=3, out_channels=16, stride=1, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(16)
        self.layer2 = DictConv2d(in_channels=16, out_channels=16, stride=2, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(16)
        self.layer3 = DictConv2d(in_channels=16, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.bn3 = BatchNorm(32)
        self.layer4 = DictConv2d(in_channels=32, out_channels=32, stride=2, kernel_size=3, padding=1)
        self.bn4 = BatchNorm(32)

        self.fc = nn.Linear(8*8*32, num_classes)

    def lenet8(self, num_classes):
        self.layer1 = DictConv2d(in_channels=3, out_channels=16, stride=1, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(16)
        self.layer2 = DictConv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(16)
        self.layer3 = DictConv2d(in_channels=16, out_channels=16, stride=2, kernel_size=3, padding=1)
        self.bn3 = BatchNorm(16)
        self.layer4 = DictConv2d(in_channels=16, out_channels=16, stride=1, kernel_size=3, padding=1)
        self.bn4 = BatchNorm(16)
        self.layer5 = DictConv2d(in_channels=16, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.bn5 = BatchNorm(32)
        self.layer6 = DictConv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.bn6 = BatchNorm(32)
        self.layer7 = DictConv2d(in_channels=32, out_channels=32, stride=2, kernel_size=3, padding=1)
        self.bn7 = BatchNorm(32)
        self.layer8 = DictConv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.bn8 = BatchNorm(32)

        # self.fc = nn.Linear(8*8*32, num_classes)
        self.fc = nn.Linear(8 * 8 * 32, num_classes)

    def update_stepsize(self):

        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

    def forward(self, x):

        out = x
        for i in range(1, self.num_layer+1):
            out = eval(f"self.bn{i}(self.layer{i}(out))")
        # out = self.bn1(self.layer1(out))
        # out = self.bn2(self.layer2(out))
        # out = self.bn3(self.layer3(out))
        # out = self.bn4(self.layer4(out))

        out = out.view(x.size(0), -1)
        y = self.fc(out)
        return y, None

    def _forward(self, x, layer_th, before_bn=False):

        # assert 1 <= layer_th <= 4
        out = x

        if layer_th == 1:
            if before_bn:
                return self.layer1(out)
            else:
                return self.bn1(self.layer1(out))
        else:
            out = self.layer1(out)
            for i in range(2, layer_th+1):
                out = eval(f"self.layer{i}(self.bn{i-1}(out))")

            if before_bn:
                return out
            else:
                return eval(f"self.bn{layer_th}(out)")

    def _inverse(self, z, layer_th, before_bn=False):

        # assert 1 <= layer_th <= 4

        if layer_th == 1:
            if before_bn:

                return self.layer1.inverse(z)
            else:
                out = self.bn1.inverse(z)
                return self.layer1.inverse(out)
        else:
            x_tilde = z
            if not before_bn:
                x_tilde = eval(f"self.bn{layer_th}.inverse(x_tilde)")

            for k in range(layer_th, 1, -1):

                x_tilde = eval(f"self.bn{k-1}.inverse(self.layer{k}.inverse(x_tilde))")

            x_tilde = self.layer1.inverse(x_tilde)

            return x_tilde

    def generate_x(self, x, layer_th, before_bn=False):
        with torch.no_grad():
            z       = self._forward(x, layer_th, before_bn)
            x_tilde = self._inverse(z, layer_th, before_bn)

            if cfg['VIZ_INPUTNORM']:
                if 'cifar' in cfg['DATASET']['DATASET']:
                    mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])[None, :, None, None].to('cuda')
                    std = torch.FloatTensor([0.2023, 0.1994, 0.2010])[None, :, None, None].to('cuda')
                elif 'imagenet' in cfg['DATASET']['DATASET']:
                    mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
                    std = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
                else:
                    raise ValueError()

                x_tilde = x_tilde * std + mean

            # normalize the image energy  NCHW
            x_max = x_tilde.amax(dim=2, keepdim=True).amax(dim=3, keepdim=True)
            x_min = x_tilde.amin(dim=2, keepdim=True).amin(dim=3, keepdim=True)
            x_norm = (x_tilde - x_min) / (x_max - x_min)

            x_hist_norm = batch_hist_equalize(x_norm.cpu()).cuda()

        return z, x_tilde, x_norm, x_hist_norm
