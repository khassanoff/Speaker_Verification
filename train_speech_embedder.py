#!/home/ykhassanov/.conda/envs/py37/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""
from __future__ import division
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
#from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from speech_embedder_net import Resnet34_VLAD, SpeechEmbedder, GE2ELoss, SILoss, get_centroids, \
get_cossim, HybridLoss

torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] =str(hp.device) # for multiple gpus

def train(model_path):
    total = []
    path1 = '/raid/saida.mussakhojayeva/projects/SEAME/dev_sge/text_lang.txt'
    path2 = '/raid/saida.mussakhojayeva/projects/SEAME/dev_man/text_lang.txt'
    path3 = '/raid/saida.mussakhojayeva/projects/SEAME/train/text_lang.txt'
    for path in [path1, path2, path3]:
        with open(path, 'r') as f:
            total += f.readlines()
    all_perm = []
    pos_en, pos_cn, pos_diff = 0, 0, 0
    print('preparing test set')
    for i, line1 in tqdm(enumerate(total)):
        line1 = line1.strip().split()
        utt_id1, start1, end1, utt1, lang1 = line1[0], line1[1], line1[2], line1[3:-1], line1[-1]
        spk_id1 = utt_id1.split('-')[0]
        for line2 in total[(i+1):]:
            line2 = line2.strip().split()
            utt_id2, start2, end2, utt2, lang2 = line2[0], line2[1], line2[2], line2[3:-1], line2[-1]
            spk_id2 = utt_id2.split('-')[0]
            if lang1==lang2:
                if lang1=='EN':lang_label = 0
                elif lang1=='CN':lang_label = 1
            else: lang_label = 2
            if spk_id1==spk_id2: label = 1
            else:label = 0
            instance = [label, utt_id1, start1, end1, utt_id2, start2, end2, lang_label," ".join(utt1), " ".join(utt2)]
            all_perm.append(instance)               
    print('loaded all permutation cases')
    
    config_values = "toy_model" + hp.model.type + "_proj" + str(hp.model.proj) + "_vlad" + str(hp.model.vlad_centers) \
                    + "_ghost" + str(hp.model.ghost_centers) + "_spk" + str(hp.train.N) + "_utt" + str(hp.train.M) \
                    + "_dropout" + str(hp.model.dropout) + "_feat" + hp.data.feat_type + "_lr" + str(hp.train.lr) \
                    + "_optim" + hp.train.optim + "_loss" + hp.train.loss \
                    + "_wd" + str(hp.train.wd) + "_fr" + str(hp.data.tisv_frame) \
                    + "_patience" + str(hp.train.patience) + "_thrsh" + str(hp.train.threshold) + "_factor" + str(hp.train.factor) 
    #checkpoint and log dir
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    log_file = config_values + ".log"
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
        loss_fn = SILoss(hp.model.proj, train_dataset.num_of_spk).cuda()
    elif hp.train.loss.lower() == 'hybrid':
        #dataset
        train_dataset = VoxCeleb()
        train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                                  num_workers=hp.train.num_workers, drop_last=True)
        loss_fn = HybridLoss(hp.model.proj, len(train_dataset)).cuda()

        
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
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True,factor=hp.train.factor, patience=hp.train.patience, threshold=hp.train.threshold)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.train.lr*0.01,
    #                cycle_momentum=False, max_lr=hp.train.lr, step_size_up=5*len(train_loader),
    #                mode="triangular")
    #print(scheduler)
    
    iteration = 0
    eer_low = 100
    for e in range(hp.train.epochs):
        step_decay(e, optimizer)       #stage based lr scheduler
        total_loss = 0
        for batch_id, (mel_db_batch, spk_id) in enumerate(train_loader):
            embedder_net.train()
            #pdb.set_trace()
            mel_db_batch = mel_db_batch.cuda()
            spk_id = spk_id.cuda()

            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2),
                                                        mel_db_batch.size(3)))
            optimizer.zero_grad()
            embeddings = embedder_net(mel_db_batch)
            #embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            #get loss, call backward, step optimizer
            loss,_ = loss_fn(embeddings, spk_id) #wants (Speaker, Utterances, embedding)
            loss.backward()
            optimizer.step()
            #scheduler.step()    #uncomment for iteration based schedulers, eg. CycliclLR
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
        
        if (e+1)%2==0: 
            #(e+1)%2==0 
            # switch model to evaluation mode
            embedder_net.eval()
            # calculate accuracy on validation set
            with torch.no_grad():
                root = "/raid/saida.mussakhojayeva/datasets/seame/"
                vectors = {}
                print('calculating embeddings')
                for index, instance in enumerate(total):
                    if '03nc06fay_0101' in instance: continue #corrupted file
                    instance = instance.split()
                    utt_id = "-".join(instance[0].split("-")[1:-2])
                    if "ni" in utt_id: utt_type='interview/'
                    elif "nc" in utt_id: utt_type = 'conversation/'
                    utt_path = root + utt_type + utt_id + "/" + "-".join(instance[0].split("-")[1:])
                    spec = np.load(utt_path+'.npy', allow_pickle=True)
                    s1 = torch.Tensor(spec).unsqueeze(0)
                    e1 = embedder_net(s1.cuda())
                    vectors[instance[0]] = e1.cpu().detach()
                print('computed all embeddings')
                scores, labels = [], []
                for instance in all_perm:
                    if '03nc06fay_0101' in instance[4] or'03nc06fay_0101' in instance[1] : continue #corrupted file
                    label = instance[0]
                    labels.append(int(label))
                    el1, el2 = instance[1], instance[4]
                    e1 = vectors[el1]
                    e2 = vectors[el2]
                    e1 = e1 / torch.norm(e1, dim=1).unsqueeze(1)
                    e2 = e2 / torch.norm(e2, dim=1).unsqueeze(1)
                    score = torch.dot(e1.squeeze(0), e2.squeeze(0)).item()
                    scores.append(score)
                print('==> computing eer')
                eer, thresh = calculate_eer(labels, np.array(scores))
                tp_en, fn_en, tn_en, fp_en = 0, 0, 0, 0
                tp_cn, fn_cn, tn_cn, fp_cn = 0, 0, 0, 0
                tp_diff, fn_diff, tn_diff, fp_diff = 0, 0, 0, 0
                for instance in all_perm:
                    lang = instance[7] # 0-en 1-cn 2-diff
                    if '03nc06fay_0101' in instance[4] or'03nc06fay_0101' in instance[1] : continue #corrupted file
                    label = int(instance[0])
                    if score > thresh:
                        pred = 1
                    else: pred = 0
                    if pred != label:
                        if pred==0: 
                            if lang==0: fn_en+=1
                            elif lang==1: fn_cn+=1
                            elif lang==2: fn_diff+=1    
                            #print('INCORRECT', len(instance[7].split()), len(instance[8].split()))
                        else: 
                            if lang==0: fp_en+=1
                            elif lang==1: fp_cn+=1
                            elif lang==2: fp_diff+=1
                    else:
                        if pred==0:
                            if lang==0: tn_en+=1
                            elif lang==1: tn_cn+=1
                            elif lang==2: tn_diff+=1
                        else:
                            if lang==0: tp_en+=1
                            elif lang==1: tp_cn+=1
                            elif lang==2: tp_diff+=1
                recall_en =  tp_en/(tp_en+fn_en)
                recall_cn = tp_cn/(tp_cn+fn_cn)
                recall_diff = tp_diff/(tp_diff+fn_diff)
                mesg = 'ENGLISH EXAMPLES: {0}, {1}, {2}, {3} '.format(tp_en, fn_en, tn_en, fp_en)
                mesg += ('(recall: %0.4f )\n'%recall_en)
                mesg += 'CHINESE EXAMPLES: {0}, {1}, {2}, {3} '.format(tp_cn, fn_cn, tn_cn, fp_cn)
                mesg += ('(recall: %0.4f )\n'%recall_cn)
                mesg += 'DIFF EXAMPLES: {0}, {1}, {2}, {3} '.format(tp_diff, fn_diff, tn_diff, fp_diff)
                mesg += ('(recall: %0.4f )\n'%recall_diff)
                print(mesg)
                with open(log_file_path, 'a') as f:
                    f.write(mesg)

                if eer < eer_low:
                    mesg = ("\n new EER : %0.4f\n"%(eer))
                    print(mesg)
                    with open(log_file_path, 'a') as f:
                        f.write(mesg)
                    if hp.train.checkpoint_dir is not None:
                        eer_low = eer
                        ckpt_model_filename = config_values + '.pth'
                        ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
                        ckpt_loss_path = os.path.join(hp.train.checkpoint_dir, 'loss_'+ckpt_model_filename)
                        torch.save(loss_fn.state_dict(), ckpt_loss_path)
                        torch.save(embedder_net.state_dict(), ckpt_model_path)
                #scheduler.step(eer) # uncommenr for ReduceLROnPlateau scheduler
                mesg = ("\nEER : %0.4f (thres:%0.2f)\n"%(eer, thresh))
                mesg += ("learning rate: {0:.8f}\n".format(optimizer.param_groups[1]['lr']))
                print(mesg)
                with open(log_file_path, 'a') as f:
                    f.write(mesg)


def testVoxCelebOptim(model_path):
    model_path = os.path.join(hp.train.checkpoint_dir, model_path)
    print('==> calculating test({}) data lists...'.format(os.path.join(hp.data.test_path,
                                                                       hp.data.feat_type)))
    #Load model
    print('==> loading model({})'.format(model_path))
    if hp.model.type.lower() == 'tresnet34':
        embedder_net = Resnet34_VLAD()
    elif hp.model.type.lower() == 'rnn':
        embedder_net = SpeechEmbedder()
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


def testJusanBank(p1, p2, model_path):
    model_path = os.path.join(hp.train.checkpoint_dir, model_path)
    embedder_net = Resnet34_VLAD()
    embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    print('==> transforming wav to spec')
    s1 = extract_all_feat(p1, mode = 'test').transpose()    #dim: time, spec
    s2 = extract_all_feat(p2, mode = 'test').transpose()
    s1 = torch.Tensor(s1).unsqueeze(0)
    s2 = torch.Tensor(s2).unsqueeze(0)
    print('==> computing vectors')
    e1 = embedder_net(s1.cuda())
    e2 = embedder_net(s2.cuda())
    e1 = e1 / torch.norm(e1, dim=1).unsqueeze(1)
    e2 = e2 / torch.norm(e2, dim=1).unsqueeze(1)
    score = torch.dot(e1.squeeze(0), e2.squeeze(0)).item()
    print(score)
    return score
    #print('==> computing eer')
    #eer, thresh = calculate_eer(labels, np.array([score]))
    #print("\nEER : %0.4f (thres:%0.2f)\n"%(eer, thresh))
    
    
    
if __name__=="__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        #testVoxCelebOptim(hp.model.model_path)
        testJusanBank(hp.data.p1, hp.data.p2, hp.model.model_path)
