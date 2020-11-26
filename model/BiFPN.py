import torch
import torch.nn as nn

from module import ConvModule, Swish, MemoryEfficientSwish
from utils import Conv2SamePadding, PoolwithPadding

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, norm=True, activation=False, onnx_export=False):
        super(SepConv, self).__init__()
        
        self.norm=norm
        self.activation=activation
        if out_ch is None:
            out_ch=in_ch

        self.depth_conv=Conv2SamePadding(in_ch, in_ch, kernel_size=3, stride=1, groups=in_ch, bias=False)
        self.point_conv=Conv2SamePadding(in_ch, out_ch, kernel_size=1, stride=1)
      
        if self.norm:
            self.bn=nn.BatchNorm2d(num_features=out_ch, momentum=0.01, eps=1e-3)

        if self.activation: 
            self.swish=MemoryEfficientSwish() if not onnx_export else Swish()


    def forward(self, x):
        x=self.depth_conv(x)
        x=self.point_conv(x)

        if self.norm:
            x=self.bn(x)
        
        if self.activation:
            x=self.swish(x)

        return x

class BiFPN(nn.Module):
    def __init__(self,):
        super(BiFPN, self).__init__()


class BiFPN_sub(nn.Module):
    def __init__(self, in_ch, out_ch, num_out, start_level=0 , end_level=1, stack=1):
        super(BiFPN, self).__init__()

        assert isinstance(in_ch, list)

        self.out_ch=out_ch
        self.num_out=num_out
        self.num_ins=len(in_ch)
        self.stack=stack

        if end_level==-1:
            self.backbone_end_level=self.num_ins
            assert num_out>=self.num_ins-start_level

        else:
            self.backbone_end_level=end_level
            assert end_level<=len(in_ch)
            assert num_out==end_level-start_level

        self.end_level=end_level
        self.start_level=start_level

        self.lateral_conv=nn.ModuleList()
        self.fpn_conv=nn.ModuleList()
        self.stack_bifpn_conv=nn.ModuleList()

    
        # Layer sacling
        self.conv6_up=SepConv(in_ch, onnx_export=True)
        self.conv5_up=SepConv(in_ch, onnx_export=True)
        self.conv4_up=SepConv(in_ch, onnx_export=True)
        self.conv3_up=SepConv(in_ch, onnx_export=True)

        self.conv7_down=SepConv(in_ch, onnx_export=True)
        self.conv6_down=SepConv(in_ch, onnx_export=True)
        self.conv5_down=SepConv(in_ch, onnx_export=True)
        self.conv4_down=SepConv(in_ch, onnx_export=True)

        # Feature scaling
        self.p6_upsample=nn.Upsample(scale_factor=2, mode='neareast') # Result of p7 upsample
        self.p5_upsample=nn.Upsample(scale_factor=2, mode='neareast')
        self.p4_upsample=nn.Upsample(scale_factor=2, mode='neareast')
        self.p3_upsample=nn.Upsample(scale_factor=2, mode='neareast')

        self.p4_downsample= PoolwithPadding()# Result of p3 downsample
        self.p5_downsample= PoolwithPadding()
        self.p6_downsample=PoolwithPadding()
        self.p7_downsample =PoolwithPadding()



        

        for i in range(self.start_level, self.backbone_end_level):
            l_conv=SepConv()

class BiFPNMoudle(nn.Module):
    def __init__(self, channel, level, init=0.5, conv_cfg=None, norm_cfg=None, activation=None, eps=1e-4)
        super(BiFPNMoudle, self).__init__()
        self.activation=activation
        self.eps=eps
        self.level=level
        self.bifpn_convs=nn.ModuleList()

        # weighted
        self.w1=nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1=nn.ReLU()
        self.w2=nn.Parameter(torch.Tensor(3, levels-2).fill_(init))
        self.relu2=nn.ReLU()

        for idx in range(2):
            for i in range(self.level-1) #1,2,3


        #weight
        self.w1=nn.Parameter(torch.Tensor(2, level))






