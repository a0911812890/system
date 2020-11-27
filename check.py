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


# target, curr_sr = librosa.load('/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/target.wav', sr=16000, mono=True)
# enhance, curr_sr = librosa.load('/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/enhance.wav', sr=16000, mono=True)
# noisy, curr_sr = librosa.load('/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/noisy.wav', sr=16000, mono=True)
# print(pesq( target, target,16000))
# print(pesq( target, enhance,16000))
# print(pesq( target, noisy,16000))
# print(pesq(16000, target, target,'wb'))
# print(pesq(16000, target, enhance,'wb'))
# print(pesq(16000, target, noisy,'wb'))

with h5py.File("/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/hdf/test.hdf5", "r") as f:
    print(f.attrs["sr"]) 
        