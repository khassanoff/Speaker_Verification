#!/home/ykhassanov/.conda/envs/py37/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
import pdb
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec

class VoxCeleb(Dataset):
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        self.path = os.path.join(hp.data.train_path, hp.data.feat_type)
        self.utter_num = hp.train.M
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        if self.shuffle:
            # select random speaker
            selected_file = random.sample(np_file_list, 1)[0]
        else:
            selected_file = np_file_list[idx]
        # load utterance spectrogram of selected speaker
        utters = np.load(os.path.join(self.path, selected_file))
        if self.shuffle:
            # select M utterances per speaker
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)
            utterance = utters[utter_index]
        else:
            # utterances of a speaker [batch(M), n_mels, frames]
            utterance = utters[self.utter_start: self.utter_start+self.utter_num]

        #utterance = utterance[:,:,:160]               # TODO implement variable length batch size
        # transpose [batch, frames, n_mels]
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))
        return utterance
