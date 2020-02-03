#!/home/ykhassanov/.conda/envs/py37/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from hparam import hparam as hp
from utils import get_centroids, get_cossim, calc_loss

class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer,
                                  batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x

class GE2ELoss(nn.Module):
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True).cuda()
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True).cuda()

    def forward(self, embeddings, y=None):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss

class SILoss(nn.Module):
    def __init__(self, emb_size, num_of_spks):
        super(SILoss, self).__init__()
        self.linear = nn.Linear(emb_size, num_of_spks)

    def forward(self, x, y):
        y = y.repeat_interleave(hp.train.M)
        x = torch.reshape(x, (hp.train.N*hp.train.M, x.size(2)))
        x = self.linear(x)
        loss = F.cross_entropy(x, y)
        return loss

class HybridLoss(nn.Module):
    def __init__(self, emb_size, num_of_spks):
        super(HybridLoss, self).__init__()
        self.ge2eloss = GE2ELoss()
        self.siloss = SILoss(emb_size, num_of_spks)

    def forward(self, x, y):
        l1 = self.ge2eloss(x)
        l2 = self.siloss(x, y)
        return l1 + l2


input_dim = 1
block1_input = 64
filters = [[48, 48, 96], [96, 96, 128], [128, 128, 256], [256, 256, 512]]
conv_block_input = 32
def conv_block(input_dim, filters, strides):
    return nn.Sequential(
            nn.Conv2d(input_dim, filters[0], kernel_size=1, bias=False, stride=strides),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], filters[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[2])
            )

    
def shortcut(input_dim, filters, strides):
    return nn.Sequential(
            nn.Conv2d(input_dim, filters[2], kernel_size=1, bias=False, stride=strides),
            nn.BatchNorm2d(filters[2])
            )
    

def identity_block(input_dim, filters):
    return nn.Sequential(
            nn.Conv2d(input_dim, filters[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(True),
            nn.Conv2d(filters[1], filters[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[2])
            )
    
    

class Resnet34_VLAD(nn.Module):
    def __init__(self):
        super(Resnet34_VLAD, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, block1_input, kernel_size=7, bias=False, padding=3)
        self.norm_layer = nn.BatchNorm2d(block1_input)
        self.relu = nn.ReLU(True)
        self.max_pool1 = nn.MaxPool2d((2, 2))
        
        self.conv_block1 = conv_block(64, filters[0], (1, 1))
        self.shortcut1 = shortcut(64, filters[0],  (1, 1))
        self.identity_block1 = identity_block(filters[0][2], filters[0])
        
        self.conv_block2 = conv_block(filters[0][2], filters[1],  (2, 2))
        self.shortcut2 = shortcut(filters[0][2], filters[1], (2, 2))
        self.identity_block2 = identity_block(filters[1][2], filters[1])
        
        self.conv_block3 = conv_block(filters[1][2], filters[2],  (2, 2))
        self.shortcut3 = shortcut(filters[1][2], filters[2], (2, 2))
        self.identity_block3 = identity_block(filters[2][2], filters[2])
        
        self.conv_block4 = conv_block(filters[2][2], filters[3],  (2, 2))
        self.shortcut4 = shortcut(filters[2][2], filters[3], (2, 2))
        self.identity_block4 = identity_block(filters[3][2], filters[3])
        self.max_pool2 = nn.MaxPool2d((3, 1), stride=(2, 1))
        
        self.conv2 = nn.Conv2d(filters[3][2], filters[3][2], kernel_size=(7,1), bias=True)
        self.conv3 = nn.Conv2d(filters[3][2], hp.model.vlad_centers+hp.model.ghost_centers, kernel_size=(7,1), bias=True)
        
        self.cluster = nn.Parameter(data=torch.Tensor(hp.model.vlad_centers+hp.model.ghost_centers, 512), requires_grad=True)
        self.dense = nn.Linear(hp.model.vlad_centers*512, hp.model.proj)
        self.dropout = nn.Dropout(hp.model.dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Parameter):
                nn.init.orthogonal_(m.weight)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        # ============================
        #            Block 1
        # ============================
        x1 = self.conv1(x)
        x1 = self.norm_layer(x1)
        x1 = self.relu(x1)
        x1 = self.max_pool1(x1) #?

        # ============================
        #            Block 2
        # ============================
        x2 = self.relu(self.conv_block1(x1).add(self.shortcut1(x1)))
        x2 = self.relu(self.identity_block1(x2).add(x2))
        
        
        # ============================
        #            Block 3
        # ============================             
        x3 = self.relu(self.conv_block2(x2).add(self.shortcut2(x2)))
        x3 = self.relu(self.identity_block2(x3).add(x3))
        x3 = self.relu(self.identity_block2(x3).add(x3))
        
        # ============================
        #            Block 4
        # ============================             
        x4 = self.relu(self.conv_block3(x3).add(self.shortcut3(x3)))
        x4 = self.relu(self.identity_block3(x4).add(x4)) 
        x4 = self.relu(self.identity_block3(x4).add(x4)) 

        
        # ============================
        #            Block 5
        # ============================             
        x5 = self.relu(self.conv_block4(x4).add(self.shortcut4(x4)))
        x5 = self.relu(self.identity_block4(x5).add(x5))
        x5 = self.relu(self.identity_block4(x5).add(x5))
        x5 = self.max_pool2(x5)
        
        # ============================
        #   Fully Connected Block 1
        # ============================   
        feat_broadcast = self.relu(self.conv2(x5)) # output (1, 512, 1, 16)
        feat_broadcast = torch.transpose(feat_broadcast, 1, -1)
        
        # ============================
        #  Feature Aggregation (VLAD)
        # ============================
        
        x_centers = self.conv3(x5)
        x_centers = torch.transpose(x_centers , 1, -1)

        #num_features = x.shape[1] #512
        max_cluster_score = torch.max(x_centers, -1, keepdim=True).values
        exp_cluster_score = torch.exp(x_centers - max_cluster_score)
        A = exp_cluster_score / torch.sum(exp_cluster_score, -1, keepdim = True)

        A = A.unsqueeze(-1)
    
        #feat_broadcast = x_fc  # feat_broadcast : bz x W x H x 1 x D
        feat_broadcast = feat_broadcast.unsqueeze(-2)
        feat_res = feat_broadcast - self.cluster
        weighted_res = torch.mul(A, feat_res)
        cluster_res  = torch.sum(weighted_res, (1, 2))
        
        if hp.model.ghost_centers>0:
            cluster_res = cluster_res[:, :hp.model.vlad_centers, :]
            
        cluster_l2 = nn.functional.normalize(cluster_res,dim=-1, p=2)
        x = cluster_l2.view(-1, hp.model.vlad_centers*512)
        
        # ============================
        #   Fully Connected Block 2
        # ============================  
        x = self.dense(x)
        x = self.dropout(x)
        
        #unit vector normalization
        #x = x / torch.norm(x, dim=1).unsqueeze(1)

        return x
