#!/home/ykhassanov/.conda/envs/py37/bin/python
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
from utils import calculate_eer 

from hparam import hparam as hp
from data_load import VoxCeleb
#from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from speech_embedder_net import Resnet34_VLAD, SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model_path):
    device = torch.device("cuda:"+str(hp.device) if torch.cuda.is_available() else "cpu")
    #device = torch.device(hp.device)
 
    #checkpoint and log dir
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    log_file = "model" + hp.model.type + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                    + "_feat" + hp.data.feat_type + "_lr" + str(hp.train.lr) \
                    + "_optim" + hp.train.optim + ".log"
    log_file_path = os.path.join(hp.train.checkpoint_dir, log_file)

    #dataset
    train_dataset = VoxCeleb()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                              num_workers=hp.train.num_workers, drop_last=True)
 
    #load model
    if hp.model.type.lower() == 'tresnet34':
        embedder_net = Resnet34_VLAD().to(device)
    elif hp.model.type.lower() == 'rnn':
        embedder_net = SpeechEmbedder().to(device)
    print(embedder_net)

    if hp.train.restore:
        #resume training
        embedder_net.load_state_dict(torch.load(model_path))

    ge2e_loss = GE2ELoss(device)
    #Both net and loss have trainable parameters

    if hp.train.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    elif hp.train.optim.lower() == 'adam':
        optimizer = torch.optim.Adam([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    elif hp.train.optim.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    print(optimizer)
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
                    factor=0.9, patience=hp.train.patience, threshold=0.001)
    print(scheduler)


    #start training
    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            mel_db_batch = mel_db_batch.to(device)

            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2),
                                                        mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))

            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
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
                    scheduler.step(total_loss)
                    print("learning rate: {0:.6f}\n".format(optimizer.param_groups[1]['lr']))

        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch" + str(e+1) + "_batchID" + str(batch_id+1) \
                                    + "_model" + hp.model.type \
                                    + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                                    + "_feat" + hp.data.feat_type \
                                    + "_lr" + str(hp.train.lr) + "_optim" + hp.train.optim + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)

            eer, thresh = testVoxCeleb(ckpt_model_path)
            mesg = ("\nEER : %0.2f (thres:%0.2f)\n"%(eer, thresh))
            mesg += ("learning rate: {0:.8f}\n".format(optimizer.param_groups[1]['lr']))
            with open(log_file_path, 'a') as f:
                f.write(mesg)
            embedder_net.to(device).train()

    #save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch" + str(e+1) + "_batchID" + str(batch_id+1) \
                            + "_model" + hp.model.type \
                            + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                            + "_feat" + hp.data.feat_type \
                            + "_lr" + str(hp.train.lr) + "_optim" + hp.train.optim + ".pth"

    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)
    testVoxCeleb(save_model_path)


def testVoxCeleb(model_path):
    device = torch.device("cuda:"+str(hp.device) if torch.cuda.is_available() else "cpu")

    print('==> calculating test({}) data lists...'.format(os.path.join(hp.data.test_path,
                                                                       hp.data.feat_type)))
    #Load model
    print('==> loading model({})'.format(model_path))
    if hp.model.type.lower() == 'tresnet34':
        embedder_net = Resnet34_VLAD().to(device)
    elif hp.model.type.lower() == 'rnn':
        embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    print(embedder_net)

    print('==> reading data')
    triplets = np.load(os.path.join(hp.data.test_path, hp.data.feat_type, 'test_triplets.npy'),
                       allow_pickle=True)
    print("*number of test cases {}".format(triplets.shape[0]))

    print('==> computing scores')
    scores, labels = [], []
    counter = 0
    for t in triplets:
        labels.append(t[0])
        s1 = torch.Tensor(t[1]).unsqueeze(0)
        s2 = torch.Tensor(t[2]).unsqueeze(0)
        e1 = embedder_net(s1.to(device))
        e2 = embedder_net(s2.to(device))
        #e1 = embedder_net(s1)
        #e2 = embedder_net(s2)
        scores.append(torch.dot(e1.squeeze(0), e2.squeeze(0)).item())
        counter += 1
        if counter % 1000 == 0:
            print("*completed {} pairs ({})".format(counter, time.ctime()))

    #Compute EER
    print('==> computing eer')
    eer, thresh = calculate_eer(labels, np.array(scores))
    print("\nEER : %0.2f (thres:%0.2f)"%(eer, thresh))
    return eer, thresh


if __name__=="__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        testVoxCeleb(hp.model.model_path)
