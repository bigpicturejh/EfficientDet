import torch
import torch.nn as nn
import math
from EfficientNet import EfficientNet


def ModelParing(model_idx):
    return {'efficientnet_b{}':'efficientdet_d{}'.format(model_idx)}

class EfficientDet(nn.Module):
    def __init__(self, BackboneType: str, CompoundScaleBiFPN: list, is_train=True, iou_th=0.5 ):
        super(EfficientDet, self).__init__()

        self.CompoundScaleBiFPN = CompoundScaleBiFPN
        self.ModelName= ModelParing(0)
        self.backbone=EfficientNet(self.CompoundScaleBiFPN)
        self.is_train=is_train
        # self.body=BiFPN(in_channel=self.backbone.contain)

        self.out_ch_test=self.backbone.contain[-5:]
        print(f"Num of BiFPN output channel : {self.out_ch_test}")

        
if __name__=="__main__":
    model=EfficientDet('Efficientnet',[1.0, 1.0, 1.0, 1.0])
    temp_input=torch.randn(1, 3, 64, 64)



        



