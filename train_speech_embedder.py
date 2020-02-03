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
from utils import calculate_eer, step_decay
from tqdm import tqdm as tqdm
from hparam import hparam as hp
from data_load import VoxCeleb, VoxCeleb_utter
#from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from speech_embedder_net import Resnet34_VLAD, SpeechEmbedder, GE2ELoss, SILoss, get_centroids, \
get_cossim, HybridLoss

torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.device) # for multiple gpus
    #device = torch.device("cuda:"+str(hp.device) if torch.cuda.is_available() else "cpu")
    #device = torch.device(hp.device)
 
    #checkpoint and log dir
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    log_file = "model" + hp.model.type + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                    + "_feat" + hp.data.feat_type + "_lr" + str(hp.train.lr) \
                    + "_optim" + hp.train.optim + "_loss" + hp.train.loss \
                    + "_wd" + str(hp.train.wd) + "_fr" + str(hp.data.tisv_frame) + ".log"
    log_file_path = os.path.join(hp.train.checkpoint_dir, log_file)

 
    #load model
    if hp.model.type.lower() == 'tresnet34':
        embedder_net = Resnet34_VLAD()
    elif hp.model.type.lower() == 'rnn':
        embedder_net = SpeechEmbedder()

    if torch.cuda.device_count() > 1:
        embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    print(embedder_net)

    if hp.train.restore:
        #resume training
        embedder_net.load_state_dict(torch.load(model_path))

    if hp.train.loss.lower() == 'ge2e':
        #dataset
        train_dataset = VoxCeleb()
        train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                                  num_workers=hp.train.num_workers, drop_last=True)
        loss_fn = GE2ELoss().cuda()
    elif hp.train.loss.lower() == 'si':
        #dataset
        train_dataset = VoxCeleb_utter()
        train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                                  num_workers=hp.train.num_workers, drop_last=True)
        loss_fn = SILoss(512, train_dataset.num_of_spk).cuda()
    elif hp.train.loss.lower() == 'hybrid':
        #dataset
        train_dataset = VoxCeleb()
        train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                                  num_workers=hp.train.num_workers, drop_last=True)
        loss_fn = HybridLoss(512, len(train_dataset)).cuda()

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
 
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True,
    #                factor=0.9, patience=hp.train.patience, threshold=0.0001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.train.lr*0.01,
                    cycle_momentum=False, max_lr=hp.train.lr, step_size_up=5*len(train_loader),
                    mode="triangular")
    print(scheduler)

    #start training
    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        #step_decay(e, optimizer)
        total_loss = 0
        for batch_id, (mel_db_batch, spk_id) in enumerate(train_loader):
            mel_db_batch = mel_db_batch.cuda()
            spk_id = spk_id.cuda()

            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2),
                                                        mel_db_batch.size(3)))
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch)
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))

            #get loss, call backward, step optimizer
            loss = loss_fn(embeddings, spk_id) #wants (Speaker, Utterances, embedding)
            loss.backward()

            optimizer.step()
            scheduler.step()    #uncomment for iteration based schedulers, eg. CycliclLR
            #print("learning rate: {0:.6f}\n".format(optimizer.param_groups[1]['lr']))

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

        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch" + str(e+1) + "_batchID" + str(batch_id+1) \
                                    + "_model" + hp.model.type \
                                    + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                                    + "_feat" + hp.data.feat_type \
                                    + "_lr" + str(hp.train.lr) + "_optim" + hp.train.optim \
                                    + "_loss" + hp.train.loss \
                                    + "_wd" + str(hp.train.wd) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)

            eer, thresh = testVoxCelebOptim(ckpt_model_path)
            mesg = ("\nEER : %0.4f (thres:%0.2f)\n"%(eer, thresh))
            mesg += ("learning rate: {0:.8f}\n".format(optimizer.param_groups[1]['lr']))
            with open(log_file_path, 'a') as f:
                f.write(mesg)
            embedder_net.train().cuda()

    #save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch" + str(e+1) + "_batchID" + str(batch_id+1) \
                            + "_model" + hp.model.type \
                            + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                            + "_feat" + hp.data.feat_type \
                            + "_lr" + str(hp.train.lr) + "_optim" + hp.train.optim \
                            + "_loss" + hp.train.loss \
                            + "_wd" + str(hp.train.wd) + ".pth"

    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)
    testVoxCelebOptim(save_model_path)


def testVoxCelebOptim(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] =str(hp.device) # for multiple gpus
    print('==> calculating test({}) data lists...'.format(os.path.join(hp.data.test_path,
                                                                       hp.data.feat_type)))
    #Load model
    print('==> loading model({})'.format(model_path))
    if hp.model.type.lower() == 'tresnet34':
        embedder_net = Resnet34_VLAD()
    elif hp.model.type.lower() == 'rnn':
        embedder_net = SpeechEmbedder()
        
    if torch.cuda.device_count() > 1:
        embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    #print(embedder_net)
    print("Model type: "+hp.model.type)
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
        #pdb.set_trace()
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
    print("\nEER : %0.4f (thres:%0.2f)\n"%(eer, thresh))
    return eer, thresh
    
    
    
def testVoxCeleb(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] =str(hp.device) # for multiple gpus
    #device = torch.device("cuda:"+str(hp.device) if torch.cuda.is_available() else "cpu")

    print('==> calculating test({}) data lists...'.format(os.path.join(hp.data.test_path,
                                                                       hp.data.feat_type)))
    #Load model
    print('==> loading model({})'.format(model_path))
    if hp.model.type.lower() == 'tresnet34':
        embedder_net = Resnet34_VLAD()
    elif hp.model.type.lower() == 'rnn':
        embedder_net = SpeechEmbedder()

    if torch.cuda.device_count() > 1:
        embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    #print(embedder_net)
    print("Model type: "+hp.model.type)

    #Load data
    print('==> reading data')
    triplet_files = os.listdir(os.path.join(hp.data.test_path, hp.data.feat_type))
    #triplets = np.load(os.path.join(hp.data.test_path, hp.data.feat_type, 'test_triplets.npy'),
    #                   allow_pickle=True)
    print("*number of test cases {}".format(len(triplet_files)))

    print('==> computing scores')
    scores, labels = [], []
    counter = 0
    for f in triplet_files:
        t = np.load(os.path.join(hp.data.test_path, hp.data.feat_type, f), allow_pickle=True)
        labels.append(t[0])
        s1 = torch.Tensor(t[1]).unsqueeze(0)
        s2 = torch.Tensor(t[2]).unsqueeze(0)
        e1 = embedder_net(s1.cuda())
        e2 = embedder_net(s2.cuda())
        #normalize
        e1 = e1 / torch.norm(e1, dim=1).unsqueeze(1)
        e2 = e2 / torch.norm(e2, dim=1).unsqueeze(1)
        #e1 = embedder_net(s1)
        #e2 = embedder_net(s2)
        scores.append(torch.dot(e1.squeeze(0), e2.squeeze(0)).item())
        counter += 1
        if counter % 1000 == 0:
            print("*completed {} pairs ({})".format(counter, time.ctime()))
    print("*completed {} pairs ({})".format(counter, time.ctime()))

    #Compute EER
    print('==> computing eer')
    eer, thresh = calculate_eer(labels, np.array(scores))
    print("\nEER : %0.4f (thres:%0.2f)"%(eer, thresh))
    return eer, thresh


if __name__=="__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        testVoxCelebOptim(hp.model.model_path)
