import torch
import torch.nn as nn
import math
import torch.nn.functional as F
 
class Conv2SamePadding(nn.Moduel):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super(Conv2dStaticPadding, self).__init__()

        self.conv=nn.Conv2d(in_ch, out_ch, kernel_size, stride, bias=bias, groups=groups)
        self.stride=self.conv.stride
        self.kernel_size=self.conv.kernel_size
        self.dilation=self.conv.dilation

        if isinstance(self.stride, int):
            self.stride=[self.stride]*2
        elif len(self.stride)==1:
            self.stride=[self.stride[0]]*2
        
        if isinstance(self.kernel, int):
            self.kernel_size=[self.kernel_size]*2
        elif len(self.kernel_size)==1:
            self.kernel_size = [self.kernel_size[0]]*2


    def forward(self, x):
        h, w= x.shape[-2:]

        extra_h=(math.ceil(w/self.stride[1])-1)*self.stride[1]-w+self.kernel_size[1]
        extra_v=(math.ceil(h/self.stride[0])-1)*self.stride[0]-h+self.kernel_size[0]

        top=extra_v//2
        bottom=extra_v-top
        left=extra_h//2
        right=extra_h-left

        x=F.pad(x, [left, right, top, bottom])
        x=self.conv(x)

        return x


class PoolwithPadding(nn.Module):
    def __init__(self, *args):
        super(PoolwithPadding, self).__init__()

        self.pool=nn.MaxPool2d(*args, **kwargs)
        self.stride=self.pool.stride
        self.kernel_size=self.pool.kernel_size
 
    def forward(self, x):

        height, width = x.shape[-2:] # batch, channel, height, width
        
        expand_height=(math.ceil(height/self.stride[0])-1)*self.stride[0]-height+self.kernel_size[1]
        expand_width=(math.ceil(width/self.stride[1])-1)*self.stride[1]-width+self.kernel_size[0]

        left_pad=expand_width//2
        right_pad=expand_width-left_pad
        top_pad=expand_height//2
        bottom_pad=expand_height-top_pad

        padded_x=F.pad(x, (left_pad, right_pad, top_pad, bottom_pad))
        return padded_x


        
        
        


if __name__=="__main__":
    tempTensor=torch.rand(1, 3, 64, 64)
