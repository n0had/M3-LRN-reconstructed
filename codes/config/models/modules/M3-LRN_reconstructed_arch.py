import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
  
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
          
class MobilenetV2_backbone(nn.Module):
    def __init__(self): #deleted width_mult
        super(MobilenetV2_backbone, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = 32#_make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = c #_make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = 1280 #_make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(output_channel, num_classes)

        #self._initialize_weights()######

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

    #def _initialize_weights(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #            m.weight.data.normal_(0, math.sqrt(2. / n))
    #            if m.bias is not None:
    #                m.bias.data.zero_()
    #        elif isinstance(m, nn.BatchNorm2d):
    #            m.weight.data.fill_(1)
    #            m.bias.data.zero_()
    #        elif isinstance(m, nn.Linear):
    #            m.weight.data.normal_(0, 0.01)
    #            m.bias.data.zero_()


#assuming the MobilenetV2 backbone isnt static?
#class MobilenetV2_backbone(nn.module):
#  def __init__(self):
#    super.__init__()
#    
#    self.body=
#    
#  def forward(self, x):
#    return body(x)
  
class z_and_3DMM(nn.module):
  def __init__(self):
    super(z_and_3DMM, self).__init__()
  
    self.MNv2=MobilenetV2_backbone()
    self.FC=nn.Linear(1280,62)
    
    #3 MNv2 for 3 heads? otherwise, what would it mean?

  def forward(self, x):
    x=self.MNv2(x)
    x=torch.cat((FC(x), x), -1)#right cat dim?
    return x
  

class z_alpha_to_refined_landmarks(nn.module):
    def __init__(self):
        super(z_alpha_to_refined_landmarks, self).__init__()
        
        self.decoder=z_and_3DMM()
        self.alphas_to_68landmarks=
        self.mlp64=
        self.mlp64_to_holistic=
        self.MMPF_to_refined_landmarks=
    
    def forward(self, x):
        x=self.decoder(x)
        alphas=x[0:61]
        y=self.alphas_to_68landmarks(alphas)
        y=self.mlp64(y)
        holistic=self.mlp64_to_holistic(y)
        alphas_no_pose=x[0:49]# ok?
        
        #which operations are ok for computing backprop? use matrices instead of cat and [0:49] and copy?
        
        vector2354=torch.cat((holistic, alphas_no_pose), -1)#right cat dim? same as z_and_3DMM.
        vector2354repeated=vector2354.repeat(1, 68, 1) #because of batchsize?
        
        MMPF=torch.cat((y, vector2354repeated), 2)#right cat dim?? carefully match dims!
        #make sure y (i.e. mlp64 output) is of shape torch.Size([1, 68, 64])
        
        refined_landmarks=self.MMPF_to_refined_landmarks(MMPF)
        refined_landmarks_1D=torch.reshape(refined_landmarks, (-1,)) # inverse action too?
        refined_landmarks_with_z_alpha=torch.cat((refined_landmarks_1D, x), -1)#right cat dim? same as z_and_3DMM. 1 or -1?
        
        return refined_landmarks_with_z_alpha

   




