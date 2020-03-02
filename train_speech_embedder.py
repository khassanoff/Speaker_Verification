# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import torch
import pdb
import numpy as np
from torch.utils.data import DataLoader
from utils import calculate_eer, step_decay, extract_all_feat
from tqdm import tqdm as tqdm
from hparam import hparam as hp
from data_load import VoxCeleb, VoxCeleb_utter
from speech_embedder_net import Resnet34_VLAD, SpeechEmbedder, GE2ELoss, SILoss, get_centroids, \
get_cossim, HybridLoss

torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] =str(hp.device) # for multiple gpus

def train(model_path):
    config_values = "model" + hp.model.type + "_proj" + str(hp.model.proj) + "_vlad" + str(hp.model.vlad_centers) \
                    + "_ghost" + str(hp.model.ghost_centers) + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                    + "_dropout" + str(hp.model.dropout) + "_feat" + hp.data.feat_type + "_lr" + str(hp.train.lr) \
                    + "_optim" + hp.train.optim + "_loss" + hp.train.loss \
                    + "_wd" + str(hp.train.wd) + "_fr" + str(hp.data.tisv_frame)
    #checkpoint and log dir
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    log_file = config_values + ".log"
    log_file_path = os.path.join(hp.train.checkpoint_dir, log_file)
    
    #load model
    embedder_net = Resnet34_VLAD()
    embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    print(embedder_net)

    #load dataset
    train_dataset = VoxCeleb_utter()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                                  num_workers=hp.train.num_workers, drop_last=True)
    loss_fn = SILoss(hp.model.proj, train_dataset.num_of_spk).cuda()
    
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(os.path.join(hp.train.checkpoint_dir, model_path)))
        loss_fn.load_state_dict(torch.load(os.path.join(hp.train.checkpoint_dir, "loss_" + model_path)))
    #Both net and loss have trainable parameters

    if hp.train.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': loss_fn.parameters()}
                ], lr=hp.train.lr, weight_decay=hp.train.wd)
    elif hp.train.optim.lower() == 'adam':
        optimizer = torch.optim.Adam([
                    {'params': embedder_net.parameters()},
                    {'params': loss_fn.parameters()}
                ], lr=hp.train.lr, weight_decay=hp.train.wd)
    elif hp.train.optim.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta([
                    {'params': embedder_net.parameters()},
                    {'params': loss_fn.parameters()}
                ], lr=hp.train.lr, weight_decay=hp.train.wd)
        
    print(optimizer)
    iteration = 0
    for e in range(hp.train.epochs):
        step_decay(e, optimizer) #stage based lr scheduler
        total_loss = 0
        
        for batch_id, (mel_db_batch, spk_id) in enumerate(train_loader):
            embedder_net.train().cuda()
            mel_db_batch = mel_db_batch.cuda()
            spk_id = spk_id.cuda()
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2),
                                                        mel_db_batch.size(3)))
            optimizer.zero_grad()
            embeddings = embedder_net(mel_db_batch)
            #get loss, call backward, step optimizer
            loss,_ = loss_fn(embeddings, spk_id) #wants (Speaker, Utterances, embedding)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
            iteration += 1
            
            if (batch_id + 1) % hp.train.log_interval == 0 or \
               (batch_id + 1) % (len(train_dataset)//hp.train.N) == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}], Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(
                        time.ctime(), e+1, batch_id+1, len(train_dataset)//hp.train.N, iteration,
                        loss, total_loss / (batch_id + 1))
                print(mesg)
                with open(log_file_path, 'a') as f:
                    f.write(mesg)
                    
                if (batch_id + 1) % (len(train_dataset)//hp.train.N) == 0:
                    #scheduler.step(total_loss) # uncommenr for ReduceLROnPlateau scheduler
                    print("learning rate: {0:.6f}\n".format(optimizer.param_groups[1]['lr']))
        
        # calculate accuracy on validation set
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            # switch model to evaluation mode
            embedder_net.eval()

            ckpt_model_filename = config_values + '.pth'
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            ckpt_loss_path = os.path.join(hp.train.checkpoint_dir, 'loss_'+ckpt_model_filename)
            torch.save(loss_fn.state_dict(), ckpt_loss_path)
            torch.save(embedder_net.state_dict(), ckpt_model_path)

            eer, thresh = testVoxCeleb(ckpt_model_path)
            mesg = ("\nEER : %0.4f (thres:%0.2f)\n"%(eer, thresh))
            mesg += ("learning rate: {0:.8f}\n".format(optimizer.param_groups[1]['lr']))
            print(mesg)
            with open(log_file_path, 'a') as f:
                f.write(mesg)


def testVoxCeleb(model_path):
    print('==> calculating test({}) data lists...'.format(os.path.join(hp.data.test_path,
                                                                       hp.data.feat_type)))
    #Load model
    print('==> loading model({})'.format(model_path))
    embedder_net = Resnet34_VLAD()
    embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    
    verify_list = np.loadtxt(hp.data.test_meta_path, str)
    list1 = np.array([i[1] for i in verify_list])
    list2 = np.array([i[2] for i in verify_list])
    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list, return_index=True)
    vectors = {}
    print('==> computing unique vectors')
    counter = 0
    for index, wav_file in tqdm(enumerate(unique_list[0])):
        original_index = unique_list[1][index]
        spec = np.load(os.path.join(hp.data.test_path, hp.data.feat_type, 'test_triplet'+str(original_index)+'.npy'), allow_pickle=True)[1]
        s1 = torch.Tensor(spec).unsqueeze(0)
        e1 = embedder_net(s1.cuda())
        vectors[wav_file] = e1.cpu().detach()
        
    scores, labels = [], []
    for (label, el1, el2) in verify_list:
        labels.append(int(label))
        e1 = vectors[el1]
        e2 = vectors[el2]
        e1 = e1 / torch.norm(e1, dim=1).unsqueeze(1)
        e2 = e2 / torch.norm(e2, dim=1).unsqueeze(1)
        scores.append(torch.dot(e1.squeeze(0), e2.squeeze(0)).item())
    #Compute EER
    print('==> computing eer')
    eer, thresh = calculate_eer(labels, np.array(scores))
    return eer, thresh

    

if __name__=="__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        testVoxCeleb(hp.model.model_path)
