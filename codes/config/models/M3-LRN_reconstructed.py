
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models import modules



def class M3_LRN_model():
  
  def __init__(self):
        self.device = torch.device("cuda")      
        
        self.network=modules.M3LRN().to(self.device)
        
        self.lambda1=0.02
        self.lambda2=0.03
        self.lambda3=0.02
        self.lambda4=0.001
        
        self.L2_loss = nn.L2Loss().to(self.device)
        self.L1_smooth_loss = nn.SmoothL1Loss().to(self.device)
        
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.08, momentum=0.9)
        
        self.LRscheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[48,64], gamma=0.1)
        
        self.total_epoches=80
        self.batch_size=1024
    
    
    train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
          
            
    def optimize_parameters(self, step):
        self.optimizer.zero_grad()
        alphas_and_z, refined_landmarks, alphas_from_landmarks = self.network(self.feed_img)

        total_loss = 0
        for ind in range(len(alphas_and_z)):
            d_3DMM = self.L2_loss(alphas_and_z[ind], self.real_alphas_and_z)
            d_lmk = self.L1_smooth_loss(refined_landmarks[ind], self.real_landmarks)
            d_3DMM_lmk=self.L2_loss(alphas_from_landmarks[ind], self.real_alphas_and_z)
            d_g=self.L2_loss(alphas_and_z[ind][0:62], alphas_from_landmarks)

        total_loss += self.lambda1*d_3DMM
        total_loss += self.lambda1*d_lmk
        total_loss += self.lambda1*d_3DMM_lmk
        total_loss += self.lambda1*d_g

        total_loss.backward()
        self.optimizer.step()

    def train_network():
      for epoch in range(1, self.total_epoches+1):
        
