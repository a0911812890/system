import h5py
import musdb
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import glob

from tqdm import tqdm

from utils import load, write_wav


def crop(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    '''
    targets = targets[:, shapes["output_start_frame"]:shapes["output_end_frame"]]
    return mix, targets

class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False):
        '''

        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    mix_audio, _ = load(example["input"], sr=self.sr, mono=(self.channels == 1))
                    
                    source_audios, _ = load(example["target"], sr=self.sr, mono=(self.channels == 1))
                    assert(source_audios.shape[1] == mix_audio.shape[1])
                    
                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels :
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel are not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]
    
            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))#每個檔案開始的位置
        self.length = self.start_pos[-1]#總長度
        #print(self.start_pos)

    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)
        
        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index) # 根據index 看是哪個音檔 回頭找到他的 audio_idx 0,1,3,5,8,10,11
        
        if audio_idx > 0: # 
            index = index - self.start_pos[audio_idx - 1] # 看是在這個音檔的哪個片段

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]
        #print(audio_length,target_length)
        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"] # 第幾個片段

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0
        #print('start_pos',start_pos)
        #print('end_pos',end_pos)
        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        target = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            target = np.pad(target, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)


        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, target = self.audio_transform(audio, target)
        # print(audio.shape,target.shape)
        return audio, target

    def __len__(self):
        return self.length

#------------------------------Myself------------------------------#
def get_ling_data_list(database_path):
    samples = list()
    noisy='noisy_testset'
    clean='clean_testset'
    noisy_path=os.path.join(database_path,noisy)
    speech_path=os.path.join(database_path,clean)
    lists=os.listdir(speech_path)
    for i in range(len(lists)):
        pair={"input" :os.path.join(noisy_path,lists[i]) ,"target" : os.path.join(speech_path,lists[i]) } 
        samples.append(pair)
    data={'noisy' : samples }
    return data
def get_enhance_folds(database_path):
    subset_types = ["test"]
    for subset_type in subset_types:
        noisy_wav='noisy'
        speech_wav='speech'
        enhance_path='/media/hd03/sutsaiwei_data/data/mydata/enhance'
        noisy_path=os.path.join(database_path,'outside_test1','noisy')
        speech_path=os.path.join(database_path,'outside_test1','speech')
        # all_path=database_path+'/'+subset+'/'+noisy_wav+'/wav/'
        samples = list()
        samples2 = list()
        shift=1
        enhance_list=os.listdir(enhance_path)
        for i in range(len(enhance_list)):
            enhance_file_name=enhance_list[i].split("_",5)
            target_path=os.path.join(speech_path,enhance_file_name[1+shift]+"_"+enhance_file_name[2+shift]+"_"+enhance_file_name[3+shift]+".wav")
            input_path=os.path.join(enhance_path,enhance_list[i])
            pair={"input" :input_path ,"target" : target_path }
            samples.append(pair)

        shift=0
        noisy_list=os.listdir(noisy_path)
        for i in range(len(noisy_list)):
            noisy_file_name=noisy_list[i].split("_",4)
            target_path=os.path.join(speech_path,noisy_file_name[1+shift]+"_"+noisy_file_name[2+shift]+"_"+noisy_file_name[3+shift]+".wav")
            input_path=os.path.join(noisy_path,noisy_list[i])
            pair={"input" :input_path ,"target" : target_path }
            samples2.append(pair)
        data={'enhance' : samples ,'noisy' : samples2 }
    # print(f'all train song:{(len(subsets[0]))} all val song:{(len(subsets[1]))}  all test song:{(len(subsets[2]))} ')
    return data
def get_folds(database_path,outside_test):
    database_path = database_path
    subsets = list()
    if outside_test:
        subset_types = ["train","outside_test", "outside_test"]
    else:
        subset_types = ["train","test", "test"]
    for subset_type in subset_types:
        noisy_wav='noisy'
        speech_wav='speech'
        noisy_path=os.path.join(database_path,subset_type,'noisy')
        speech_path=os.path.join(database_path,subset_type,'speech')
        # all_path=database_path+'/'+subset+'/'+noisy_wav+'/wav/'
        samples = list()
        noisy_list=os.listdir(noisy_path)
        for i in range(len(noisy_list)):
            noisy_file_name=noisy_list[i].split("_",4)
            target_path=os.path.join(speech_path,noisy_file_name[1]+"_"+noisy_file_name[2]+"_"+noisy_file_name[3]+".wav")
            input_path=os.path.join(noisy_path,noisy_list[i])
            pair={"input" :input_path ,"target" : target_path }
            samples.append(pair)
        subsets.append(samples)
    data={'train' : subsets[0],'val' : subsets[1],  'test' : subsets[2]}
    # print(f'all train song:{(len(subsets[0]))} all val song:{(len(subsets[1]))}  all test song:{(len(subsets[2]))} ')
    return data

