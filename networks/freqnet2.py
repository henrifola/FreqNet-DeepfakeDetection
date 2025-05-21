import torch.nn as nn
from torch.nn import functional as F
import torch

__all__ = ['FreqNet', 'freqnet']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class FreqNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=1, zero_init_residual=False):
        super(FreqNet, self).__init__()

        # Pre-FFT conv layers
        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)).cuda())
        self.bias1   = nn.Parameter(torch.randn((64,)).cuda())
        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)).cuda())
        self.bias2   = nn.Parameter(torch.randn((64,)).cuda())
        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias3   = nn.Parameter(torch.randn((256,)).cuda())
        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias4   = nn.Parameter(torch.randn((256,)).cuda())

        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

        # Color conv layers
        self.crl_weight1 = nn.Parameter(torch.randn(16, 3, 1, 1).cuda())
        self.crl_bias1 = nn.Parameter(torch.randn(16).cuda())
        self.crl_weight2 = nn.Parameter(torch.randn(32, 16, 1, 1).cuda())
        self.crl_bias2 = nn.Parameter(torch.randn(32).cuda())
        self.crl_weight3 = nn.Parameter(torch.randn(3, 32, 1, 1).cuda())
        self.crl_bias3 = nn.Parameter(torch.randn(3).cuda())

        # Shared spectral convs
        self.fcl_amp = conv1x1(3, 3)
        self.fcl_phase = conv1x1(3, 3)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def crl(self, x):
        x = F.relu(F.conv2d(x, self.crl_weight1, self.crl_bias1))
        x = F.relu(F.conv2d(x, self.crl_weight2, self.crl_bias2))
        x = F.relu(F.conv2d(x, self.crl_weight3, self.crl_bias3))
        return x

    def hfreqWH(self, x, scale):
        x = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=[-2, -1])
        b,c,h,w = x.shape
        x[:,:,h//2-h//scale:h//2+h//scale,w//2-w//scale:w//2+w//scale] = 0
        x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-2, -1]), norm="ortho")
        return F.relu(torch.real(x))

    def hfreqC(self, x, scale):
        x = torch.fft.fftshift(torch.fft.fft(x, dim=1, norm="ortho"), dim=1)
        b,c,h,w = x.shape
        x[:,c//2-c//scale:c//2+c//scale,:,:] = 0
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        return F.relu(torch.real(x))

    def fcl_block(self, x):
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=[-2, -1])
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        amp_feat = F.relu(self.fcl_amp(amp))
        phase_feat = F.relu(self.fcl_phase(phase))
        real = amp_feat * torch.cos(phase_feat)
        imag = amp_feat * torch.sin(phase_feat)
        x_re = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(real, imag), dim=[-2, -1]), norm="ortho")
        return F.relu(torch.real(x_re))

    def forward(self, x):
        
        x = self.crl(x)
        x = self.hfreqWH(x, 4)
        x = F.relu(F.conv2d(x, self.weight1, self.bias1))
        x = self.hfreqC(x, 4)
        x = self.fcl_block(x)

        x = self.hfreqWH(x, 4)
        x = F.relu(F.conv2d(x, self.weight2, self.bias2, stride=2))
        x = self.hfreqC(x, 4)
        x = self.fcl_block(x)

        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.hfreqWH(x, 4)
        x = F.relu(F.conv2d(x, self.weight3, self.bias3))
        x = self.fcl_block(x)

        x = self.hfreqWH(x, 4)
        x = F.relu(F.conv2d(x, self.weight4, self.bias4, stride=2))
        x = self.fcl_block(x)

        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)

def freqnet(**kwargs):
    return FreqNet()
