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
from typing import Tuple
class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 middle_channels: Tuple[int, int, int],
                 kernel_size,
                 stride: Tuple[int, int] = (2, 2),
                 use_shortcut: bool = True):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels[0], kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(middle_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(middle_channels[0], middle_channels[1], kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(middle_channels[1], middle_channels[2], kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(middle_channels[2]),
        )
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels[2], kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(middle_channels[2]),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block(x)
        out = self.block1(out)
        out = self.block2(out)

        out += self.shortcut(identity)

        out = self.activation(out)

        return out
    
    
class Decoder(nn.Module):

    def __init__(self,
                 middle_channels: int,
                 out_channels: int,
                 k_clusters: int = 8,
                 g_clusters: int = 2,
                 mode: str = 'gvlad',
                 use_attention: bool = False):
        super(Decoder, self).__init__()

        self.k_clusters = k_clusters
        self.mode = mode
        self.g_clusters = g_clusters

        if self.mode != 'gvlad':
            self.g_clusters = 0

        self.attention = nn.Identity()

        if use_attention:
            self.attention = ConvolutionalBlockAttentionModule(middle_channels, kernel_size=7)

        self.conv = nn.Sequential(nn.Conv2d(middle_channels, middle_channels, kernel_size=(7, 1), stride=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.conv_center = nn.Conv2d(middle_channels, self.k_clusters + self.g_clusters, kernel_size=(7, 1), stride=(1, 1))

        self.vlad_polling = VladPooling(middle_channels, self.k_clusters, self.g_clusters, self.mode)

        self.fc = nn.Sequential(nn.Linear(middle_channels * self.k_clusters, middle_channels, bias=True),
                                nn.ReLU(inplace=True))

        self.logit = nn.Linear(middle_channels, out_channels, bias=False)

    def forward(self, features, **kwargs):

        x5, x4, x3, x2, x1 = features

        features = self.attention(x5)

        conv = self.conv(features)

        conv_center = self.conv_center(features)

        embeddings = self.vlad_polling((conv, conv_center))

        embeddings = self.fc(embeddings)

        logit = self.logit(embeddings)

        return embeddings, logit

class SILoss(nn.Module):
    def __init__(self):
        super(SILoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.ce(x, y)

class HybridLoss(nn.Module):
    def __init__(self, emb_size, num_of_spks):
        super(HybridLoss, self).__init__()
        self.ge2eloss = GE2ELoss()
        self.siloss = SILoss(emb_size, num_of_spks)

    def forward(self, x, y):
        l1 = self.ge2eloss(x)
        l2 = self.siloss(x, y)
        return l1 + l2

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
                
class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, features):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(features)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x     
    
    
class SpeakerRecognition(EncoderDecoder):

    def __init__(
            self,
            middle_channels: int,
            out_channels: int,
            k_clusters: int = 8,
            g_clusters: int = 2,
            encoder_name: str = 'resnet34s',
            encoder_weights: str = None,
            mode: str = 'gvlad',
            use_attention: bool = False,
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = Decoder(middle_channels,
                          out_channels,
                          k_clusters,
                          g_clusters,
                          mode,
                          use_attention)

        super().__init__(encoder, decoder, None)

        super().initialize()

        self.name = 'sp--{0}--{1}'.format(mode, encoder_name)    
    


class ResNet34s(nn.Module):
    def __init__(self, in_channels: int,
                 middle_channels: int):
        super(ResNet34s, self).__init__()
        self.in_channels = in_channels

        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=(7, 7), padding=2,  bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.x2 = nn.Sequential(
            ConvBlock(middle_channels, (48,  48, 96), kernel_size=3, stride=(1, 1)),
            ConvBlock(96, (48,  48, 96), kernel_size=3, use_shortcut=False, stride=(1, 1)),
        )

        self.x3 = nn.Sequential(
            ConvBlock(96, (96, 96, 128), kernel_size=3),
            ConvBlock(128, (96, 96, 128), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            ConvBlock(128, (96, 96, 128), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
        )

        self.x4 = nn.Sequential(
            ConvBlock(128, (128, 128, 256), kernel_size=3),
            ConvBlock(256, (128, 128, 256), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            ConvBlock(256, (128, 128, 256), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
        )

        # stride=(2, 1) or stride=(2, 2) ?
        self.x5 = nn.Sequential(
            ConvBlock(256, (256, 256, 512), kernel_size=3),
            ConvBlock(512, (256, 256, 512), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            ConvBlock(512, (256, 256, 512), kernel_size=3, stride=(1, 1),
                      use_shortcut=False),
            nn.MaxPool2d((3, 1), stride=(2, 2))
        )

    def forward(self, x):
        #pdb.set_trace()
        x = x.unsqueeze(1)
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        return [x5, x4, x3, x2, x1]    
    
resnet_encoders = {
    'resnet34s': {
        'encoder': ResNet34s,
        'out_shapes': (512, 256, 128, 96, 64),
        'params': {
            'in_channels': 1,
            'middle_channels': 64,
        },
    }
}

encoders = {}
encoders.update(resnet_encoders)    


class VladPooling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 k_clusters: int = 8,
                 g_clusters: int = 2,
                 mode: str = 'gvlad'):

        super(VladPooling, self).__init__()
        self.k_clusters = k_clusters
        self.mode = mode
        self.g_clusters = g_clusters

        self.centroids = nn.Parameter(torch.randn(k_clusters + g_clusters, in_channels), requires_grad=True)
        self._init_params()

    def _init_params(self):
        nn.init.orthogonal_(self.centroids.data)

    def forward(self, x):

        features, cluster_score = x

        num_features = features.shape[1]

        max_cluster_score = cluster_score.max(dim=1, keepdim=True)[0]

        soft_assign = F.softmax(cluster_score - max_cluster_score, dim=1)

        residual_features = features.unsqueeze(1)

        residual_features = residual_features - self.centroids.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        soft_assign = soft_assign.unsqueeze(2)

        weighted_res = soft_assign * residual_features

        cluster_res = torch.sum(weighted_res, dim=(3, 4), keepdim=False)

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_clusters, :]

        cluster_l2 = F.normalize(cluster_res, p=2, dim=-1)
        outputs = cluster_l2.reshape(-1, (int(self.k_clusters) * int(num_features)))

        return outputs



def get_encoder(name, encoder_weights=None):
    Encoder = encoders[name]['encoder']

    encoder = Encoder(**encoders[name]['params'])

    encoder.out_shapes = encoders[name]['out_shapes']


    return encoder    
    
    
