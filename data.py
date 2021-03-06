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
        audio_idx = self.start_pos.bisect_right(index) # 根據index 看是哪個音檔 回頭找到他的 audio_idx
        
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
            start_target_pos = index * self.shapes["output_frames"]

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

        # targets = {inst : targets[idx*self.channels:(idx+1)*self.channels] for idx, inst in enumerate(self.instruments)}

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, target = self.audio_transform(audio, target)
        # print(audio.shape,target.shape)
        return audio, target

    def __len__(self):
        return self.length

#------------------------------Myself------------------------------#
def get_folds(database_path):
    path = database_path
    subsets = list()
    for subset in ["train","val", "test"]:
        noisy_wav='noisy'
        speech_wav='speech'
        all_path=os.path.join(path,subset,noisy_wav,'wav')
        # all_path=path+'/'+subset+'/'+noisy_wav+'/wav/'
        samples = list()
        noisy_list=os.listdir(all_path)
        for i in range(len(noisy_list)):
            t=noisy_list[i].split("_",4)
            target_path=os.path.join(path,subset,speech_wav,'wav',t[1]+"_"+t[2]+"_"+t[3]+".wav")
            # target_path=path+'/'+subset+'/'+speech_wav+'/wav/'+t[1]+"_"+t[2]+"_"+t[3]+".wav"
            pair={"input" :os.path.join(all_path,noisy_list[i]),"target" : target_path }
            samples.append(pair)
        subsets.append(samples)
    data={'train' : subsets[0],'val' : subsets[1],  'test' : subsets[2]}
    print(f'all train song:{(len(subsets[0]))} all val song:{(len(subsets[1]))}  all test song:{(len(subsets[2]))} ')
    return data

#------------------------------unuse------------------------------#
# def getMUSDBHQ(database_path):
#     subsets = list()

#     for subset in ["train", "test"]:
#         print("Loading " + subset + " set...")
#         tracks = glob.glob(os.path.join(database_path, subset, "*"))
#         samples = list()

#         # Go through tracks
#         for track_folder in sorted(tracks):
#             # Skip track if mixture is already written, assuming this track is done already
#             example = dict()
#             for stem in ["mix", "bass", "drums", "other", "vocals"]:
#                 filename = stem if stem != "mix" else "mixture"
#                 audio_path = os.path.join(track_folder, filename + ".wav")
#                 example[stem] = audio_path

#             # Add other instruments to form accompaniment
#             acc_path = os.path.join(track_folder, "accompaniment.wav")

#             if not os.path.exists(acc_path):
#                 print("Writing accompaniment to " + track_folder)
#                 stem_audio = []
#                 for stem in ["bass", "drums", "other"]:
#                     audio, sr = load(example[stem], sr=None, mono=False)
#                     stem_audio.append(audio)
#                 acc_audio = np.clip(sum(stem_audio), -1.0, 1.0)
#                 write_wav(acc_path, acc_audio, sr)

#             example["accompaniment"] = acc_path

#             samples.append(example)

#         subsets.append(samples)

#     return subsets

# def getMUSDB(database_path):
#     mus = musdb.DB(root=database_path, is_wav=False)

#     subsets = list()

#     for subset in ["train", "test"]:
#         tracks = mus.load_mus_tracks(subset)
#         samples = list()

#         # Go through tracks
#         for track in tracks:
#             # Skip track if mixture is already written, assuming this track is done already
#             track_path = track.path[:-4]
#             mix_path = track_path + "_mix.wav"
#             acc_path = track_path + "_accompaniment.wav"
#             if os.path.exists(mix_path):
#                 print("WARNING: Skipping track " + mix_path + " since it exists already")

#                 # Add paths and then skip
#                 paths = {"mix" : mix_path, "accompaniment" : acc_path}
#                 paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

#                 samples.append(paths)

#                 continue

#             rate = track.rate

#             # Go through each instrument
#             paths = dict()
#             stem_audio = dict()
#             for stem in ["bass", "drums", "other", "vocals"]:
#                 path = track_path + "_" + stem + ".wav"
#                 audio = track.targets[stem].audio
#                 write_wav(path, audio, rate)
#                 stem_audio[stem] = audio
#                 paths[stem] = path

#             # Add other instruments to form accompaniment
#             acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
#             write_wav(acc_path, acc_audio, rate)
#             paths["accompaniment"] = acc_path

#             # Create mixture
#             mix_audio = track.audio
#             write_wav(mix_path, mix_audio, rate)
#             paths["mix"] = mix_path

#             diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
#             print("Maximum absolute deviation from source additivity constraint: " + str(np.max(diff_signal)))# Check if acc+vocals=mix
#             print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

#             samples.append(paths)

#         subsets.append(samples)

#     print("DONE preparing dataset!")
#     return subsets

# def get_musdb_folds(root_path):
#     dataset = getMUSDB(root_path)
#     train_val_list = dataset[0]
#     test_list = dataset[1]
    
#     np.random.seed(1337)
#     train_list = np.random.choice(train_val_list, 75, replace=False)
#     val_list = [elem for elem in train_val_list if elem not in train_list]
#     print("First training song: " + str(train_list[0]))
#     #print(f'train_list \n {train_list}')
#     return {"train" : train_list, "val" : val_list, "test" : test_list}

# def random_amplify(mix, targets, shapes, min, max):
#     '''
#     Data augmentation by randomly amplifying sources before adding them to form a new mixture
#     :param mix: Original mixture
#     :param targets: Source targets
#     :param shapes: Shape dict from model
#     :param min: Minimum possible amplification
#     :param max: Maximum possible amplification
#     :return: New data point as tuple (mix, targets)
#     '''
#     residual = mix  # start with original mix
#     for key in targets.keys():
#         if key != "mix":
#             residual -= targets[key]  # subtract all instruments (output is zero if all instruments add to mix)
#     mix = residual * np.random.uniform(min, max)  # also apply gain data augmentation to residual
#     for key in targets.keys():
#         if key != "mix":
#             targets[key] = targets[key] * np.random.uniform(min, max)
#             mix += targets[key]  # add instrument with gain data augmentation to mix
#     mix = np.clip(mix, -1.0, 1.0)
#     return crop(mix, targets, shapes)
