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
import pdb, re
from random import shuffle, sample
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import load_feat

class VoxCeleb(Dataset):
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        self.path = os.path.join(hp.data.train_path, hp.data.feat_type)
        self.utter_num = hp.train.M
        self.spk_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        #spk_list = os.listdir(self.path)

        #if self.shuffle:
            # select random speaker
        #    selected_file = random.sample(spk_list, 1)[0]
        #else:
        #    selected_file = spk_list[idx]
        selected_spk = self.spk_list[idx]
        spk_id = int(re.findall("(\d+)", selected_spk)[0])

        # load utterance spectrogram of selected speaker
        utters = os.listdir(os.path.join(self.path, selected_spk))
        #utters = np.load(os.path.join(self.path, selected_spk))
        if self.shuffle:
            # select M utterances per speaker
            utters_sample = sample(utters, self.utter_num)
            #utter_index = np.random.randint(0, len(utters), self.utter_num)
            #utterance = utters[utter_index]
        else:
            # utterances of a speaker [batch(M), n_mels, frames]
            utters_sample = utters[self.utter_start: self.utter_start+self.utter_num]

        utterance = []
        for utt in utters_sample:
            utterance.append(load_feat(os.path.join(self.path, selected_spk, utt)))

        utterance = np.array(utterance)
        #utterance = utterance[:,:,:160]               # TODO implement variable length batch size
        # transpose [batch, frames, n_mels]
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))
        return utterance, spk_id

class VoxCeleb_utter(Dataset):
    #utterance based itteation (used for speaker identification task)
    def __init__(self, shuffle=True, utter_start=0):
        # data path
        self.path = os.path.join(hp.data.train_path, hp.data.feat_type)
        self.num_of_spk = len(os.listdir(self.path))
        self.utt_list = glob.glob(self.path+'/*/*')
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        selected_utt = self.utt_list[idx]
        spk_id = int(re.findall("(\d+)", os.path.basename(os.path.dirname(selected_utt)))[0])
        utterance = load_feat(selected_utt)
        utterance = np.array(utterance)
        utterance = np.expand_dims(utterance, axis=0)
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))
        return utterance, spk_id


class VoxCeleb_code2(Dataset):
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

        #if self.shuffle:
            # select random speaker
        #    selected_file = random.sample(np_file_list, 1)[0]
        #else:
        #    selected_file = np_file_list[idx]
        selected_file = np_file_list[idx]
        spk_id = int(re.findall("(\d+)", selected_file)[0])

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
        return utterance, spk_id
