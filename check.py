import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import utils
import os
import librosa
from utils import compute_loss
import soundfile
from pypesq import pesq
import pandas as pd
from pystoi import stoi
# mix_audio2, mix_sr = librosa.load('audio_examples/Cristina Vane - So Easy/mix.mp3', sr=None, mono=False)
# mix_audio, mix_sr = librosa.load('audio_examples/Cristina Vane - So Easy/mix.mp3', sr=None, mono=False)
# mix_channels = mix_audio.shape[0]
# mix_len = mix_audio.shape[1]
# print(mix_channels,mix_len,mix_audio.shape())
# utils.write_wav('test.wav', mix_audio, mix_sr)
#soundfile.write('audio_examples/Cristina Vane - So Easy/mix.mp3', audio, sr, "PCM_16")
# y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

# path = '/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/timit_10'
# subsets = list()
# for subset in ["train", "test"]:
#     noisy_wav='noisy'
#     speech_wav='speech'
#     all_path=path+'/'+subset+'/'+noisy_wav+'/wav/'
#     samples = list()
#     noisy_list=os.listdir(all_path)
#     for i in range(10):
#         t=noisy_list[i].split("_",4)
#         target_path=path+'/'+subset+'/'+speech_wav+'/wav/'+t[1]+"_"+t[2]+"_"+t[3]+".wav"
#         pair={"input" :all_path+noisy_list[i],"target" : target_path }
#         samples.append(pair)
#     subsets.append(samples)
# data={'train' : subsets[0], 'test' : subsets[1]}
# print(data['test'])
    # print(len(y),curr_sr)
    # print(len(y2),curr_sr2)
    # print(np.array_equal(y, y2))

# a='/media/hd03/sutsaiwei_data/data/mydata/train/speech/DR1_FETB0_SX68.wav'
# b='/media/hd03/sutsaiwei_data/data/mydata/train/noisy/destroyerops_DR1_FETB0_SX68_10.wav'
# c='/media/hd03/sutsaiwei_data/data/mydata/train/noisy/destroyerops_DR1_FETB0_SX68_-10.wav'
# target, curr_sr = librosa.load(os.path.join(a), sr=16000, mono=True)
# enhance, curr_sr = librosa.load(os.path.join(b), sr=16000, mono=True)
# noisy, curr_sr = librosa.load(os.path.join(c), sr=16000, mono=True)

# t=np.stack((target,target),0)
# s=np.stack((enhance,enhance),0)
# # d = stoi(target, noisy, 16000, extended=False)
# # d2 = stoi(target, enhance, 16000, extended=False)
# # print(d,d2)
# print(pesq( t, s,16000))
# print(pesq( target, noisy,16000))

from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
import random
import os
path='/media/hd03/sutsaiwei_data/data/mydata/outside_test/'
speech_file = os.listdir(path+'speech')
noise_file = os.listdir(path+'noise')
with tqdm(total=4*len(speech_file)*len(noise_file)) as pbar:
    for i in speech_file:
        for j in noise_file:
            for snr in [-7.5,-2.5,2.5,7.5]:
                speech, a_sr = librosa.load(path+'speech/'+i, sr=16000)
                noise, b_sr = librosa.load(path+'noise/'+j, sr=16000)
                start = random.randint(0, noise.shape[0] - speech.shape[0])
                n_b = noise[start:start+speech.shape[0]]


                sum_s = np.sum(speech ** 2)
                sum_n = np.sum(n_b ** 2)
                x = np.sqrt(sum_s/(sum_n * pow(10, snr/10)))
                after_noise = x * n_b
                target = speech + after_noise
                output_file=f'{path}noisy/{j[:-4]}_{i[:-4]}_{snr}.wav'
                sf.write(output_file, target, 16000)
                pbar.update(1)