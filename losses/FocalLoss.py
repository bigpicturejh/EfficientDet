import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, input, target, anchors,alpha, gamma=2.0):
        super(FocalLoss, self).__init__()

        
        _, width, height=input.size()
        classificationLoss=[]
        
        for anchor in anchors:
            anchor_X, anchor_Y=anchor[0:1]
            anchor_width, anchor_height=anchor[2:3]


            # modulating factor (1-p)^r
            guide=torch.ones(anchor_width, anchor_height)
            p=IoU(input, target)
            sub_modulator=guidance-p
            
            modulator=torch.pow(modulator, gamma )

            # Final loss for classification
            classLoss=-modulator*torch.log(p)
            classificationLoss.append(classLoss)





        


