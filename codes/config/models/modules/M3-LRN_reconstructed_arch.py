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

class Linear_decoder(nn.module):
  def __init__(self, input_dim,*dims):
        super(self).__init__()
        dims.insert(0, input_dim)
        
        FC_layers = [nn.linear(dims[k], dims[k+1]) for k in range(len(dims)-1)]
        self.decoder=nn.sequential(*FC_layers)
    
    def forward(self, x):
        return decoder(x)
        
        
        
  
class Z_and_3DMM(nn.module):
  def __init__(self):
    super(z_and_3DMM, self).__init__()
  
    self.encoder=MobilenetV2_backbone()
    self.shape_decoder=Linear_decoder(1280,320,160,40)
    self.expression_decoder=Linear_decoder(1280,320,80,10)
    self.position_decoder=Linear_decoder(1280,320,80,12)
    #self.FC=nn.Linear(1280,62)
    
    #3 MNv2 for 3 heads? otherwise, what would it mean?

  def forward(self, img):
    z=self.encoder(img)
    alphas_z=torch.cat((self.shape_decoder(z), self.expression_decoder(z), self.position_decoder(z), z), -1)#right cat dim?
    return alphas_z #first come 3DMM parameters, then z.

class MLP64_module(nn.module):
    def __init__(self):
        super(MLP64, self).__init__()
        
        self.sharedMLP=nn.sequential(nn.linear(3,64), nn.ReLU(), nn.BatchNorm1d(num_features=64), nn.linear(64,64), nn.ReLU(), nn.BatchNorm1d(num_features=64))
    
    def forward(self, x):
        #for index in range x.shape[2]:
        #    #treat as batch?
        #    sharedMLP(torch.flatten(x[:,:,index], 1))
        y=torch.stack((self.sharedMLP(torch.flatten(x[:,:,index], 1)) for index in range x.shape[2]), 2)
        
        return y
   

class Single_MLP_layer(nn.module):
    #def __init__(self, dim_tuple):
    def __init__(self, input_dim, output_dim):
        #super(MLP_single, self).__init__()
        super(self).__init__()
        
        self.single_layer=nn.sequential(nn.linear(input_dim,output_dim), nn.ReLU(), nn.BatchNorm1d(num_features=output_dim))
    
    def forward(self, x):     
        return self.single_layer(x)

class MLP_sequence(nn.module):
    #def __init__(self, dim_tuple):
    def __init__(self, input_dim,*dims):
        #super(MLP_sequence, self).__init__()
        super(self).__init__()
        dims.insert(0, input_dim)
        layers = [Single_MLP_layer(dims[k], dims[k+1]) for k in range(len(dims)-1)]
        self.sharedMLP=nn.sequential(*layers)
    
    def forward(self, x):
        #for index in range x.shape[2]:
        #    #treat as batch?
        #    sharedMLP(torch.flatten(x[:,:,index], 1))
        y=torch.stack((self.sharedMLP(torch.flatten(x[:,:,index], 1)) for index in range x.shape[2]), 2)
        
        return y

class Z_alphas_to_refined_landmarks(nn.module):
    def __init__(self):
        super(Z_alphas_to_refined_landmarks, self).__init__()
        
        #self.decoder=z_and_3DMM()
        
        
        self.alphas_to_68landmarks=
        
        
        self.MLP64=MLP_sequence(3,64,64)
        #self.MLP64_to_holistic=nn.sequential(MLP_sequence(64, ,64, 128, 1024), maxpool...) # twice 64 in input or not?
        self.MLP64_to_preholistic=MLP_sequence(64, ,64, 128, 1024) # twice 64 in input or not?
        self.MMPF_to_refined_landmarks=MLP_sequence(2418, 512, 256, 128, 3)
    
    def forward(self, x):
        
        #x=self.decoder(x)
        
        alphas=x[0:61]
        coarse_landmarks=self.alphas_to_68landmarks(alphas)
        y=self.MLP64(coarse_landmarks)
        #holistic=self.MLP64_to_holistic(y)
        holistic=torch.flatten(F.max_pool2d(self.MLP64_to_preholistic(y),(68,1)),1)
        alphas_no_pose=x[0:49]# ok?
        
        #which operations are ok for computing backprop? use matrices instead of cat and [0:49] and copy?
        
        vector2354=torch.cat((holistic, alphas_no_pose), -1)#right cat dim? same as z_and_3DMM.
        vector2354repeated=vector2354.repeat(1, 68, 1) #because of batchsize?
        
        MMPF=torch.cat((y, vector2354repeated), 2)#right cat dim?? carefully match dims!
        #make sure y (i.e. mlp64 output) is of shape torch.Size([1, 68, 64])
        
        refined_landmarks=self.MMPF_to_refined_landmarks(MMPF)
        refined_landmarks_sc=coarse_landmarks+refined_landmarks # correct sc?
        
        
        # #flatten? #torch.flatten(refined_landmarks_sc, 1)
        # refined_landmarks_sc_1D=torch.reshape(refined_landmarks_sc, (-1,)) # inverse action too?
        
        # #refined_landmarks_with_z_alpha=torch.cat((refined_landmarks_1D, x), -1)#right cat dim? same as z_and_3DMM. 1 or -1?
        # refined_landmarks_with_alpha=torch.cat((refined_landmarks_sc_1D, alphas), -1)
        
        # #dont pass z forward?
        # return refined_landmarks_with_alpha
        return refined_landmarks_sc
    
    
class Refined_landmarks_to_alphas(nn.module):
    def __init__(self):
        super(Refined_landmarks_to_alphas, self).__init__()
        
        self.MLP_layers=MLP_sequence(3, 64, 64, 128, 256, 1024)
        
        self.shape_decoder=Linear_decoder(1024,256,128,40)
        self.expression_decoder=Linear_decoder(1024,256,64,10)
        self.position_decoder=Linear_decoder(1024,256,64,12)
        # "Later separate FC layers as converters transformthe holistic landmark features to 3DMM parameters" (section 3.3)
        #how many FC layers? why seperate?
        #self.FC_layers = nn.sequential(nn.linear(1024,256), nn.linear(256,62))
        
    def forward(self, x):
        
        y=self.MLP_layers(x)
        hls=torch.flatten(F.max_pool2d(y,(68,1)),1) #hls means hollistic_landmark_features
        #alphas=self.FC_layers(hollistic_landmark_features)
        alphas=torch.cat((self.shape_decoder(hls), self.expression_decoder(hls), self.position_decoder(hls)), -1)
        return alphas
    
    
class M3LRN(nn.module):
    def __init__(self):
        super(M3LRN, self).__init__()
        
        #self.self.decoder=Z_and_3DMM()
        self.self.img_to_z_3DMM=Z_and_3DMM()
        self.decoder_to_refined_landmarks=Z_alphas_to_refined_landmarks()
        self.landmarks_to_alphas =Refined_landmarks_to_alphas()
        
    def forward(self, img):
        
        alphas_and_z=self.img_to_z_3DMM(img)
        refined_landmarks=self.decoder_to_refined_landmarks(alphas_and_z)
        alphas_from_landmarks=self.landmarks_to_alphas(refined_landmarks)
        return [alphas_and_z, refined_landmarks, alphas_from_landmarks]

   




