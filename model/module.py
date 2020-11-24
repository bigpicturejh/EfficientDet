import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F


def build_norm_layer(cfg, num_features, postfix=''):
    # cfg: input_l
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_=cfg.copy()

    layer_type=cfg_.pop('type')
    if layer_type not in conv_cfg:
        pass
    else:
        conv_layer=conv_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name=abbr+str(postfix)

    requires_grad=cfg_.pop('requires_grad',True)
    cfg_.setdefault('eps', 1e-5)

    return layer


conv_cfg=
{
'Conv':nn.Conv2d,
'ConvWS': ConvWS2d,

}

def build_conv_layer(cfg, *args, **kwargs):
    # Retrun the type definition of layer in BiFPN

    if cfg is None:
        cfg_=dict(type='conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_=cfg.copy()

    layer_type=cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Error!')

    else:
        conv_layer=conv_cfg[layer_type]

    layer=conv_layer(*args, **kwargs, **cfg_)

    return layer

class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)*x

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward()

    @staticmethod
    def backward()

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)




class ConvModule(nn.Module):    

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto',
                 conv_cfg=None, norm_cfg=None, activation='relu', inplace=True, order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order

        assert isinstance(order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = activation is not None

        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('Do not use bias and normalization at the same time')

        self.conv = build_conv_layer(conv_cfg, in_ch, out_ch, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups,  bias=bias)

        self.in_ch=self.conv.in_ch
        self.out_ch=self.conv.out_ch
        self.kernel_size=self.conv.kernel_size
        self.stride=self.conv.stride
        self.padding=self.conv.padding
        self.dilation=self.conv.dilation
        self.groups=self.conv.groups
        self.ouput_padding=self.conv.output_padding
        self.transposed=self.conv.transposed

        # build normalization layers
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_ch=out_ch
            else:
                norm_ch=in_ch
            self.norm_name, norm=build_norm_layer(norm_cfg, norm_ch)
            self.add_module(self.norm_name, norm)

        #build activation layer
        if self.with_activation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(self.activation))
            if self.activation=='relu':
                self.activation=nn.ReLU(inplace=inplace)





def conv_ws_2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-5):
    ch_in=weight.size(0)
    weight_flat=weight.view(ch_in, -1)
    mean=weight_flat.mean(dim=1, keepdim=True).view(ch_in, 1, 1, 1)
    std=weight_flat.std(dim=1, keepdim=True).view(ch_in, 1, 1, 1)
    weight=(weight-mean)/(std+eps)

    return F.conv2d(input, weight, bias, stride, padding, dilation=dilation ,groups=groups)


class ConvWS2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-5):
        super(ConvWS2d, self).__ini__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias )
        self.eps=eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.eps)





