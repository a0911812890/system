import museval
from tqdm import tqdm
import utils
import os
import numpy as np
import torch
import random
from pypesq import pesq
from utils import compute_loss
from pystoi import stoi
def predict(audio, model):
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    # audio.shape[1] 語音長度
    expected_outputs = audio.shape[1] #預期輸出
    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"] # 模型每次輸出大小
    pad_back = audio.shape[1] % output_shift # 長度不夠時pad
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0) # 補在後面
    target_outputs = audio.shape[1]
    
    outputs = np.zeros(audio.shape, np.float32) # outputs=最後輸出
    
    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"] # 原模型會使得大小便小 所以需要pad
    
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
    print('完成',audio.shape[1])
    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):

            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]] # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            Predict_output = utils.compute_output(model, curr_input) #
            outputs[:,target_start_pos:target_start_pos+model.shapes["output_frames"]]=Predict_output.squeeze(0).cpu().numpy() #


    outputs = outputs[:,:expected_outputs]# 切回原來大小

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs

def predict_song(args, audio_path, model):
    model.eval()

    # Load mixture in original sampling rate
    mix_audio, mix_sr = utils.load(audio_path, sr=16000, mono=True)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        mix_audio = np.mean(mix_audio, axis=0, keepdims=True)
    else:
        if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [args.channels, 1])
        else:
            assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = utils.resample(mix_audio, mix_sr, args.sr)

    sources = predict(mix_audio, model)

    # Resample back to mixture sampling rate in case we had model on different sampling rate
    sources = utils.resample(sources, args.sr, mix_sr)

    # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
    diff = sources.shape[1] - mix_len
    if diff > 0:
        print("WARNING: Cropping " + str(diff) + " samples")
        sources = sources[:, :-diff]
    elif diff < 0:
        print("WARNING: Padding output by " + str(diff) + " samples")
        sources = np.pad(sources, [(0,0), (0, -diff)], "constant", 0.0)

    # Adapt channels
    if mix_channels > args.channels:
        assert(args.channels == 1)
        # Duplicate mono predictions
        sources = np.tile(sources, [mix_channels, 1])
    elif mix_channels < args.channels:
        assert(mix_channels == 1)
        # Reduce model output to mono
        sources = np.mean(sources, axis=0, keepdims=True)

    sources = np.asfortranarray(sources) # So librosa does not complain if we want to save it

    return sources

def evaluate(args, dataset, model):
    dB_list_pesq = {'-10' : list() ,'-5' : list() ,'0' : list() ,'5' : list() ,'10' : list() }
    dB_list_name_pesq = {'-10' : list() ,'-5' : list() ,'0' : list() ,'5' : list() ,'10' : list() }

    dB_list_stoi = {'-10' : list() ,'-5' : list() ,'0' : list() ,'5' : list() ,'10' : list() }
    dB_list_name_stoi = {'-10' : list() ,'-5' : list() ,'0' : list() ,'5' : list() ,'10' : list() }
    model.eval()
    with torch.no_grad():
        for example in dataset:
            print("Evaluating " + example["input"])

            # Load source references in their original sr and channel number
            target_sources = utils.load(example['target'], sr=16000, mono=True)[0].flatten()
            input_sources = utils.load(example['input'], sr=16000, mono=True)[0].flatten()

            # Predict using mixture
            pred_sources  = predict_song(args, example["input"], model).flatten()
            # print(f'type : target_sources:{type(target_sources)} pred_sources:{type(pred_sources)}')
            # print(f'shape : target_sources:{target_sources.shape} pred_sources:{pred_sources.shape} ')
            output_folder = args.output
            file_name=os.path.basename(example['input'])
            utils.write_wav(os.path.join(output_folder,'enhance_'+file_name), pred_sources.T, args.sr)
            fname,ext = os.path.splitext(file_name)
            text=fname.split("_",4)
            # Evaluate pesq
            input_pesq=round(pesq(target_sources, input_sources,16000),2)
            enhance_pesq=round(pesq(target_sources, pred_sources ,16000),2)

            # Evaluate stoi
            input_stoi = stoi(target_sources, input_sources, 16000, extended=False)
            enhance_stoi = stoi(target_sources, pred_sources, 16000, extended=False)
            # print(f'input_pesq:{input_pesq} enhance_pesq:{enhance_pesq} improve_pesq:{enhance_pesq-input_pesq} ')
            
            dB_list_pesq[text[4]].append([input_pesq,enhance_pesq,enhance_pesq-input_pesq])
            dB_list_name_pesq[text[4]].append([[input_pesq,enhance_pesq,enhance_pesq-input_pesq],example['input']])

            dB_list_stoi[text[4]].append([input_stoi,enhance_stoi,enhance_stoi-input_stoi])
            dB_list_name_stoi[text[4]].append([[input_stoi,enhance_stoi,enhance_stoi-input_stoi],example['input']])


        pesq_avg = 0
        stoi_avg = 0

        for key, value in dB_list_pesq.items():
            avg_pesq=np.mean(value,0)
            pesq_list=[[avg_pesq[0],avg_pesq[1],avg_pesq[2]],"avg_pesq"]
            dB_list_name_pesq[key].append([pesq_list])
            pesq_avg+=avg_pesq[1]

        for key, value in dB_list_stoi.items():
            avg_stoi=np.mean(value,0)
            stoi_list=[[avg_stoi[0],avg_stoi[1],avg_stoi[2]],"avg_stoi"]
            dB_list_name_stoi[key].append([stoi_list])
            stoi_avg+=avg_stoi[1]
    dB_list_name_pesq['avg']=pesq_avg/5
    dB_list_name_stoi['avg']=stoi_avg/5
    print(f'pesq_avg:{pesq_avg/5} stoi_avg:{stoi_avg/5}')
    return {'pesq' : dB_list_name_pesq ,'stoi' : dB_list_name_stoi}


def validate(args, model, criterion, test_data,writer,state):
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)
    # pesq_test=(random.randrange(len(test_data)))
    # VALIDATE
    model.eval()
    # print('np.zeros(len(test_data) // args.batch_size)',np.zeros(len(test_data) // args.batch_size))
    len_data=len(test_data) // args.batch_size
    avg_input_pesq=np.zeros(len_data+1)
    avg_enhance_pesq=np.zeros(len_data+1)
    avg_improve_pesq=np.zeros(len_data+1)

    avg_input_stoi=np.zeros(len_data+1)
    avg_enhance_stoi=np.zeros(len_data+1)
    avg_improve_stoi=np.zeros(len_data+1)
    total_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, targets) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                targets = targets.cuda()

            outputs, avg_loss = compute_loss(model, x, targets, criterion)
            total_loss += (1. / float(example_num + 1)) * (avg_loss - total_loss)

            input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0)
            
            target=torch.mean(targets[0], 0).cpu().numpy()
            pred=torch.mean(outputs[0], 0).detach().cpu().numpy()
            inputs=input_centre.cpu().numpy()
            # stoi
            values1 = stoi(target, inputs, 16000, extended=False)
            values2 = stoi(target, pred, 16000, extended=False)
            if(~np.isnan(values1) and  ~np.isnan(values2) ):
                avg_input_stoi[example_num]=values1
                avg_enhance_stoi[example_num]=values2
                avg_improve_stoi[example_num]=values2 - values1

            # pesq
            values1=round(pesq(target, inputs,16000),2)
            values2=round(pesq(target, pred ,16000),2)
            if(~np.isnan(values1) and  ~np.isnan(values2) ):
                avg_input_pesq[example_num]=values1
                avg_enhance_pesq[example_num]=values2
                avg_improve_pesq[example_num]=values2 - values1

            # print(values1,values2,values2 -values1)
            pbar.set_description("Current loss: {:.4f}".format(total_loss))
            pbar.update(1)

    print(f'val_input_pesq={np.nanmean(avg_input_pesq)}')
    print(f'val_enhance_pesq={np.nanmean(avg_enhance_pesq)}')
    print(f'val_improve_pesq={np.nanmean(avg_improve_pesq)}')

    print(f'val_input_stoi={np.nanmean(avg_input_stoi)}')
    print(f'val_enhance_stoi={np.nanmean(avg_enhance_stoi)}')
    print(f'val_improve_stoi={np.nanmean(avg_improve_stoi)}')

    writer.add_scalar("val_enhance_pesq", np.nanmean(avg_enhance_pesq), state["epochs"])
    writer.add_scalar("val_improve_pesq", np.nanmean(avg_improve_pesq), state["epochs"])

    writer.add_scalar("val_enhance_stoi", np.nanmean(avg_enhance_stoi), state["epochs"])
    writer.add_scalar("val_improve_stoi", np.nanmean(avg_improve_stoi), state["epochs"])
    
    return total_loss