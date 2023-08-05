import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchstat import stat

class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = nn.functional.hardsigmoid(y, inplace=True)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()

        self.ghost = GhostModule(in_planes,growth_rate)

    def forward(self, x):
        out = self.ghost(x)
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate, reduction, num_classes):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3),  # 通道数
            nn.BatchNorm2d(num_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dense1 = self._make_dense_layers(num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.se1 = SELayer(num_planes)
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.se2 = SELayer(num_planes)
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.se3 = SELayer(num_planes)
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.se4 = SELayer(num_planes)
        self.trans4 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(num_planes, num_classes)

        self.cnndnn = CNNDNN()


    def _make_dense_layers(self, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(Bottleneck(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.basic_conv(x)
        out = self.trans1(self.se1(self.dense1(out)))
        out1 = self.cnndnn(out)
        out = self.trans2(self.se2(self.dense2(out)))
        out = self.trans3(self.se3(self.dense3(out)))
        out = self.trans4(self.se4(self.dense4(out)))
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out2 = self.linear(out)
        return out2

# CNNBiGRU模型
class CNNDNN(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        # 1. 模型
        self.cnn = nn.Sequential(  # fbank卷积层
            nn.Conv2d(in_channels=80, out_channels=10, padding=1, kernel_size=3, stride=2, bias=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.dnn = nn.Sequential(
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.dnn(out)
        return out


def GhoDenNet():
    return DenseNet([3, 4, 8, 6], growth_rate=32, reduction=0.5, num_classes=3)#6, 12, 24, 16



if __name__ == '__main__':
    net = GhoDenNet()
    # torch.save(net, 'ghoden')
    stat(net, (3, 128, 192))






