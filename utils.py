#!/home/ykhassanov/.conda/envs/py37/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018

@author: harry
"""
import librosa
import numpy as np
import torch
import torch.autograd as grad
import torch.nn.functional as F
import pdb
from scipy.signal import lfilter, cheby2

from hparam import hparam as hp

def step_decay(epoch, optimizer):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = hp.train.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = hp.train.epochs

    if hp.train.warmup_epochs>0:
        milestone = [1, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [0.1, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = hp.train.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break

    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    for param in optimizer.param_groups:
        param['lr'] = np.float(lr)

    #return np.float(lr)

def get_centroids(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def get_cossim(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0)).cuda()
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim


def calc_loss(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1)).cuda()
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized


def load_feat(np_file, mode = 'train'):
    spec = np.load(np_file)
    if mode.lower() == 'train':
        if np.random.random() > 0.3:
            spec = spec[:, :spec.shape[1]//2]
            spec = np.concatenate((spec, spec), axis=1)
        # randomly select portion of a utterance of size tisv_frame, e.g. 250 frames
        randtime = np.random.randint(0, spec.shape[1]-hp.data.tisv_frame)
        spec = spec[:, randtime:randtime+hp.data.tisv_frame]

    return spec

def cheby_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2(order, 20, [lowcut, highcut], btype='bandpass', fs=fs)
    y = lfilter(b, a, data)
    return y

def extract_all_feat(wav_file, mode = 'train'):
    #extract and save features
    sound_file, _ = librosa.core.load(wav_file, hp.data.sr)
    if hp.data.feat_type=='spec_sr_tel': 
        fs = hp.data.sr
        lowcut = 500.0
        highcut = 5000.0
        sound_file = cheby_bandpass_filter(sound_file, lowcut, highcut, fs)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)

    #if mode.lower() == 'train':
    #    sound_file = np.append(sound_file, sound_file)
    #else:
    #    sound_file = np.append(sound_file, sound_file[::-1])
    sound_file = np.append(sound_file, sound_file[::-1])
    spec = librosa.stft(np.asfortranarray(sound_file), n_fft=hp.data.nfft, hop_length=hop_length,
                        win_length=400)
    spec = np.abs(spec)  # energy

    mu = np.mean(spec, 0, keepdims=True)
    std = np.std(spec, 0, keepdims=True)
    norm_spec = (spec - mu)/(std + 1e-5)

    if hp.data.feat_type.startswith('spec'):
        return norm_spec

    spec **= 2  # power
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, spec)  #mel spectogram

    #log_mel_spec = 10*np.log10(mel_spec + 1e-6)           #log mel spectrogram of utterances
    log_mel_spec = librosa.core.power_to_db(mel_spec)   #log mel spectogram

    return log_mel_spec


def extract_feat(wav_file, mode = 'Train', extended = False):
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)

    if mode == 'Train':
        sound_file = np.append(sound_file, sound_file)
        if np.random.random() < 0.3:
            sound_file = sound_file[::-1]
    else:
        sound_file = np.append(sound_file, sound_file[::-1])

    spec = librosa.stft(np.asfortranarray(sound_file), n_fft=hp.data.nfft, hop_length=hop_length,
                        win_length=window_length)
    spec = np.abs(spec)  # energy

    if mode == 'Train':
        # randomly select portion of a utterance of size tisv_frame, e.g. 250 frames
        randtime = np.random.randint(0, spec.shape[1]-hp.data.tisv_frame)
        spec = spec[:, randtime:randtime+hp.data.tisv_frame]


    #Mean-Var normalization
    mu = np.mean(spec, 0, keepdims=True)
    std = np.std(spec, 0, keepdims=True)
    norm_spec = (spec - mu)/(std + 1e-5)

    if hp.data.feat_type.startswith('spec'):
        return norm_spec

    spec **= 2  # power
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, spec)  #mel spectogram

    #log_mel_spec = 10*np.log10(mel_spec + 1e-6)           #log mel spectrogram of utterances
    log_mel_spec = librosa.core.power_to_db(mel_spec)   #log mel spectogram

    # Above operations are equivalent to (extracting directly from the entire raw file):
    # mel_spec = librosa.feature.melspectrogram(sound_file, sr=hp.data.sr, n_fft=hp.data.nfft,
    # hop_length = hop_length, n_mels=hp.data.nmels, win_length = window_length)

    #The Google's way
    #if False:
    #    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
        # Cut silence and fix length
    #    if wav_process == True:
    #        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length,
    #                                                 hop_length=hop_length)
    #        length = int(hp.data.sr * duration)
    #        sound_file = librosa.util.fix_length(sound_file, length)
    #    
    #    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length,
    #                        win_length=window_length)

    return log_mel_spec


def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
    return mfccs, mel_db, mag_db

if __name__ == "__main__":
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))
    embeddings = torch.tensor([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]).to(torch.float).reshape(3,2,3)
    centroids = get_centroids(embeddings)
    cossim = get_cossim(embeddings, centroids)
    sim_matrix = w*cossim + b
    loss, per_embedding_loss = calc_loss(sim_matrix)
