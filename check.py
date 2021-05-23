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
from data import  SeparationDataset,  crop ,get_folds,get_enhance_folds,get_ling_data_list
from torch.utils.data import Dataset
import time
import pysepm,nussl
#####################################################################################################################
input_list=[]
target_list=[]
inputs=np.random.rand(1,16009)
input_data=nussl.AudioSignal(audio_data_array=inputs,sample_rate=16000)
target=np.random.rand(1,16009)
target_data=nussl.AudioSignal(audio_data_array=target,sample_rate=16000)
input_sources = nussl.AudioSignal('/media/hd03/sutsaiwei_data/data/mydata/check/buccaneer1_DR1_FDAC1_SI1474_-7.5.wav')
target_sources= nussl.AudioSignal('/media/hd03/sutsaiwei_data/data/mydata/check/DR1_FDAC1_SI1474.wav')
evaluator=nussl.evaluation.BSSEvalScale(target_data,input_data )
scores = evaluator.evaluate()

print(scores)
# print(scores.path_to_input_file)
print(scores['source_0']['SI-SDR'][0])
# print(scores[target_data.path_to_input_file]['SI-SDR'][0])


# dataset = get_ling_data_list('/media/hd03/sutsaiwei_data/data/mydata/ling_data')
# length = len(dataset['noisy'])
# print(length)
# all_1=np.zeros([length])
# all_2=np.zeros([length])
# for i,example in enumerate(tqdm(dataset['noisy'])):
#     print(example['input'],example['target'])
#     input_sources,_ = librosa.load(example['input'], sr=16000)
#     target_sources,_ = librosa.load(example['target'], sr=16000)
#     input_pesq=pesq(target_sources, input_sources,16000)
#     all_1[i]=input_pesq
# print(np.mean(all_1))
################################################################################################

# dataset = get_enhance_folds('/media/hd03/sutsaiwei_data/data/mydata')
# length = len(dataset['noisy'])
# print(length)
# all_1=np.zeros([length])
# all_2=np.zeros([length])
# for i,example in enumerate(tqdm(dataset['noisy'])):
#     print(example['input'],example['target'])
#     input_sources,_ = librosa.load(example['input'], sr=16000)
#     target_sources,_ = librosa.load(example['target'], sr=16000)
#     input_pesq=pesq(target_sources, input_sources,16000)
#     all_1[i]=input_pesq
# print(np.mean(all_1))

# for i,example in enumerate(tqdm(dataset['enhance'])):
#     target_sources = utils.load(example['target'], sr=16000, mono=True)[0].flatten()
#     input_sources = utils.load(example['input'], sr=16000, mono=True)[0].flatten()
#     input_pesq=pesq(target_sources, input_sources,16000)
#     all_2[i]=input_pesq
# print(np.mean(all_2))


########################################################################################################
# class myDataset(Dataset):
#     def __init__(self, dataset,sr,channels):
#         self.data=[]
#         self.sr = sr
#         self.channels = channels
#         self.length = len(dataset)
#         super(myDataset, self).__init__()
#         for idx, example in enumerate(tqdm(dataset)):
#             # Load mix
#             mix_audio = utils.load(example["input"], sr=self.sr, mono=(self.channels == 1))[0]
#             source_audios = utils.load(example["target"], sr=self.sr, mono=(self.channels == 1))[0]
#             temp = {'inputs' : None, 'targets' : None ,'file_name' : None}
#             temp['inputs']=(mix_audio)
#             temp['targets']=(source_audios)
#             temp['file_name']=(example["input"])
#             self.data.append(temp)
#     def __getitem__(self, index):
#         return self.data[index]['inputs'], self.data[index]['targets'], self.data[index]['file_name']
#     def __len__(self):
#         return  self.length
# dataset_list = get_folds('/media/hd03/sutsaiwei_data/data/mydata',True)['test']
# Dataset = myDataset(dataset_list,16000,1)
# dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=8, worker_init_fn=utils.worker_init_fn,pin_memory=True)
# all_1=np.zeros([len(dataloader)])

# with tqdm(total=len(dataloader)) as pbar:
#     for example_num, (x, targets,file_names) in enumerate(dataloader): 
#         # print(x.shape)
#         # print(targets.shape)
#         input_pesq=pesq(x.flatten(), targets.flatten(),16000) 
#         all_1[example_num]=input_pesq
#         print(input_pesq)
#         pbar.update()
# print(f'avg = {np.mean(all_1)}')






########################################################################################################



# from tqdm import tqdm
# import numpy as np
# import soundfile as sf
# import librosa
# import random
# import os
# path='/media/hd03/sutsaiwei_data/data/mydata/outside_test/'
# speech_file = os.listdir(path+'speech')
# noise_file = os.listdir(path+'noise')
# with tqdm(total=4*len(speech_file)*len(noise_file)) as pbar:
#     for i in speech_file:
#         for j in noise_file:
#             for snr in [-7.5,-2.5,2.5,7.5]:
#                 speech, a_sr = librosa.load(path+'speech/'+i, sr=16000)
#                 noise, b_sr = librosa.load(path+'noise/'+j, sr=16000)
#                 start = random.randint(0, noise.shape[0] - speech.shape[0])
#                 n_b = noise[start:start+speech.shape[0]]


#                 sum_s = np.sum(speech ** 2)
#                 sum_n = np.sum(n_b ** 2)
#                 x = np.sqrt(sum_s/(sum_n * pow(10, snr/10)))
#                 after_noise = x * n_b
#                 target = speech + after_noise
#                 output_file=f'{path}noisy/{j[:-4]}_{i[:-4]}_{snr}.wav'
#                 sf.write(output_file, target, 16000)
#                 pbar.update(1)