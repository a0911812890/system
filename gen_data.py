from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
import random
import os

dataset_name='test'
path='/media/hd03/sutsaiwei_data/data/mydata/'+dataset_name+'/'
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