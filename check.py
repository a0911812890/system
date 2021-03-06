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


# path='/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/audio_examples/400000'
# target, curr_sr = librosa.load(os.path.join(path,'target.wav'), sr=16000, mono=True)
# enhance, curr_sr = librosa.load(os.path.join(path,'enhance.wav'), sr=16000, mono=True)
# noisy, curr_sr = librosa.load(os.path.join(path,'noisy.wav'), sr=16000, mono=True)
# d = stoi(target, noisy, 16000, extended=False)
# d2 = stoi(target, enhance, 16000, extended=False)
# print(d,d2)
# # print(type(pesq( target, target,16000)))
# print(pesq( target, enhance,16000))
# print(pesq( target, noisy,16000))
# print(pesq(16000, target, target,'wb'))
# print(pesq(16000, target, enhance,'wb'))
# print(pesq(16000, target, noisy,'wb'))
# x = float('nan')
# a=[x,1]
# print(pd.isnull(a))



from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator

for summary in tf.train.summary_iterator("/path/to/log/file"):