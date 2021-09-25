
import os

from collections import OrderedDict

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
        
        lr_scheduler.MultiStepLR_Restart(
                            self.optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
    

        if self.is_train:
            train_opt = opt["train"]
            # self.init_model() # Not use init is OK, since Pytorch has its owen init (by default)
            self.netG.train()

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_restorer = []
            optim_estimator = []
            for (k, v) in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    if "Restorer" in k:
                        optim_restorer.append(v)
                    elif "Estimator" in k:
                        optim_estimator.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))
                        
                        
            
            
            
                [
                    {"params": optim_restorer, "lr": train_opt["lr_G"]},
                    {"params": optim_estimator, "lr": train_opt["lr_E"]},
                ],
                weight_decay=wd_G,
                betas=(train_opt["beta1"], train_opt["beta2"]),
            )
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            
            else:
                print("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()
            
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
        
        
    def _set_lr(self, lr_groups_l):
        """set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
