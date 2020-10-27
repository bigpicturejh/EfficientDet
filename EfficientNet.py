import torch
import torch.nn as nn
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

params={
    'efficientnet_b0':(1.0, 1.0, 224, 0.2),
    'efficientnet_b1':(1.0, 1.1, 240, 0.2),
    'efficientnet_b2':(1.1, 1.2, 260, 0.3),
    'efficientnet_b3':(1.2, 1.4, 300, 0.3),
    'efficientnet_b4':(1.4, 1.8, 300, 0.4),
    'efficientnet_b5':(1.6, 2.2, 300, 0.4),
    'efficientnet_b6':(1.8, 2.6, 300, 0.5),
    'efficientnet_b7':(2.0, 3.1, 300, 0.5),
}

model_urls = {
    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',
    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',
    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',
    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',
    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',
    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',
    'efficientnet_b6': None,
    'efficientnet_b7': None,
}

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, kernel_size, stride=1, reduction_ratio=4, drop_connect_ratio=0.2):
        super(MBConvBlock, self).__init__()

        self.drop_connect = drop_connect_ratio
        self.use_resblcok = in_ch == out_ch and stride == 1

        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        mid_ch = in_ch*expand_ratio

        # Used in SqueezeExcitation
        reduced_ch = max(1, int(in_ch/reduction_ratio))

        layers = []

        if in_ch != mid_ch:
            layers += [ConvBNReLU(in_ch, mid_ch, 1)]

        layers += [
            ConvBNReLU(mid_ch, mid_ch, kernel_size, stride=1, groups=mid_ch),
            SqueezeExcitation(mid_ch, reduced_ch),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ]

        self.module = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keepProb = 1.0-self.drop_connect
        batchSize = x.size(0)
        randTensor = keepProb
        randTensor += torch.rand(batchSize, 1, 1, 1, device=x.device)
        binaryTensor = randTensor.floor()

        return x.div(keepProb)*binaryTensor

    def forward(self, x):
        if self.use_resblcok:
            return x + self._drop_connect(self.module(x))
        else:
            return self.module(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1):
        padding = self.pading(kernel_size, stride)

        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            Swish()
        )

    def pading(self, kernel_size, stride):

        # kernel_size=5
        # stride=1
        # padSize=4
        padSize = max(0, int(kernel_size - stride))
        # [Left, Right, Top, Bottom]
        return [padSize//2, padSize-padSize//2, padSize//2, padSize-padSize//2]


class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, reduced_ch):
        super(SqueezeExcitation, self).__init__()

        # self.in_channel=in_channel
        self.module = nn.Sequential
        (
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, reduced_ch, 1),
            Swish(),
            nn.Conv2d(reduced_ch, in_channel, 1),
            nn.Sigmoid(),

        )

    def forward(self, x):
        return self.module(x)*x


# Take a channel wise expand ratio and adjust a value to divide
def round_filter(filter, width_mul):
    if width_mul==1.0:
        return filter
    else:
        return int(make_divisible(filter*width_mul))

# Take a depth wise expand ratio and adjust a value to divide
def round_repeater(depth, depth_mul):
    if depth_mul==1.0:
        return depth
    else:
        return int(make_divisible(depth*depth_mul))


def make_divisible(value, divider=8):
    newVal=max(divider, int(value+divider/2) // divider * divider)
    if newVal < 0.9*value:
        newVal+=divider

    return newVal

class EfficientNet(nn.Module):
    def __init__(self, CompoundScale: list, dropout_ratio=0.2,num_classes=100 ): # CompoundScale ->[width_mul, depth_mul]
        super(EfficientNet, self).__init__()

        # expandRatio(mid_ch), channel, numLayer, stride, kernel
        B0_architecture=[
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 1, 3],
            [6, 112, 3, 2, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
            ]

        out_ch=round_filter(32, CompoundScale[0])
        in_ch=out_ch
        sub_module=[ConvBNReLU(in_ch, out_ch, 3, 2)]

        
        for e, c, n, s, k in B0_architecture:
            out_ch=round_filter(c, CompoundScale[0])
            repeat=round_repeater(n, CompoundScale[1])

            for i in range(repeat):
                stride=s if i==0 else 1
                sub_module+=[MBConvBlock(in_ch, out_ch, e, k, stride)]
                in_ch=out_ch
        
        last_ch=round_filter(1280, CompoundScale[0])
        sub_module+=[ConvBNReLU(in_ch, last_ch, 1)]
        
        self.module=nn.Sequential(*sub_module)
        self.classifier=nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_ch, num_classes)
        )


        #weight initializer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out=m.weight.size(0)
                init_range=1.0/math.sqrt(fan_out)
                nn.init.uniform(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    

    def forward(self, x):
        x=self.module(x)
        x=x.mean([2,3])
        x=self.classifier(x)

        return x


def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mul, depth_mul, _, dropout_ratio=params[arch]
    model=EfficientNet(compoundScale, dropout_ratio, **kwargs)


    if pretrained:
        state_dict=load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strit=False)
    return model







