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
import nussl
#import nussl
def pad_data(audio, model):
    # audio.shape[1] 語音長度
    expected_outputs = audio.shape[1] #預期輸出
    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"] # 模型每次輸出大小
    pad_back = audio.shape[1] % output_shift # 長度不夠時
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0) # 補在後面
    target_outputs = audio.shape[1]
    
    outputs = np.zeros(audio.shape, np.float32) # outputs=最後輸出
    
    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"] # 原模型會使得大小便小 所以需要pad
    
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
    # print('完成',audio.shape[1])
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
    pad_back = audio.shape[1] % output_shift # 長度不夠時
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0) # 補在後面
    target_outputs = audio.shape[1]
    
    outputs = np.zeros(audio.shape, np.float32) # outputs=最後輸出
    
    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"] # 原模型會使得大小便小 所以需要pad
    
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
    # print('完成',audio.shape[1])
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
    dB_list_pesq=dict();dB_list_name_pesq=dict()
    dB_list_stoi=dict();dB_list_name_stoi=dict()
    dB_list_SISDR=dict();dB_list_name_SISDR=dict()
    if args.outside_test :
        for i in ['-7.5' ,'-2.5' ,'2.5' ,'7.5' ]:
            dB_list_pesq[i] = list()
            dB_list_name_pesq[i] = list()

            dB_list_stoi[i] = list()
            dB_list_name_stoi[i] = list()

            dB_list_SISDR[i] = list()
            dB_list_name_SISDR[i] = list()
            test_noise_file="outside_test/noise"
    else:
        for i in ['-7.5' ,'-2.5' ,'2.5' ,'7.5' ]:
            dB_list_pesq[i] = list()
            dB_list_name_pesq[i] = list()

            dB_list_stoi[i] = list()
            dB_list_name_stoi[i] = list()

            dB_list_SISDR[i] = list()
            dB_list_name_SISDR[i] = list()
            test_noise_file="test/noise"
    noise_dir=os.path.join(args.dataset_dir,test_noise_file)
    noise_file = os.listdir(noise_dir)
    dB_noise_pesq = {}
    for i in noise_file:
        dB_noise_pesq[os.path.splitext(i)[0]]=list()
    
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset)) as pbar:
            for example in dataset:
                # Load source references in their original sr and channel number
                input_data = nussl.AudioSignal(example['input'])
                target_data = nussl.AudioSignal(example['target'])
                
                # Predict using mixture
                pred_sources  = predict_song(args, example["input"], model).flatten()

                file_name=os.path.basename(example['input'])
                
                utils.write_wav(os.path.join(args.output,'enhance_'+file_name), pred_sources.T, args.sr)
                fname,ext = os.path.splitext(file_name)
                text=fname.split("_",4)
                # Evaluate pesq
                input_sources=input_data.audio_data.flatten()
                target_sources=target_data.audio_data.flatten()

                input_pesq=pesq(target_sources, input_sources,16000)
                enhance_pesq=pesq(target_sources, pred_sources ,16000)
                # Evaluate stoi
                input_stoi = stoi(target_sources, input_sources, 16000, extended=False)
                enhance_stoi = stoi(target_sources, pred_sources, 16000, extended=False)
                # scores[target_sources.path_to_input_file]['SI-SDR'][0]
                enhance_data=nussl.AudioSignal(audio_data_array=pred_sources,sample_rate=16000)
                evaluator=nussl.evaluation.BSSEvalScale(target_data  ,input_data )
                scores = evaluator.evaluate()
                input_SISDR=scores[target_data.path_to_input_file]['SI-SDR'][0]
                evaluator=nussl.evaluation.BSSEvalScale(target_data  ,enhance_data )
                scores = evaluator.evaluate()
                enhance_SISDR=scores[target_data.path_to_input_file]['SI-SDR'][0]
                

                filename=os.path.basename(example['input'])
                noise_name=filename.split("_")[0]
                dB_noise_pesq[noise_name].append([input_pesq,enhance_pesq,enhance_pesq-input_pesq,enhance_SISDR,enhance_SISDR-input_SISDR])
                
                dB_list_pesq[text[4]].append([input_pesq,enhance_pesq,enhance_pesq-input_pesq])
                dB_list_name_pesq[text[4]].append([[input_pesq,enhance_pesq,enhance_pesq-input_pesq],file_name])

                dB_list_stoi[text[4]].append([input_stoi,enhance_stoi,enhance_stoi-input_stoi])
                dB_list_name_stoi[text[4]].append([[input_stoi,enhance_stoi,enhance_stoi-input_stoi],file_name])

                dB_list_SISDR[text[4]].append([input_SISDR,enhance_SISDR,enhance_SISDR-input_SISDR])
                dB_list_name_SISDR[text[4]].append([[input_SISDR,enhance_SISDR,enhance_SISDR-input_SISDR],file_name])
                pbar.update(1)
        num=len(dB_list_pesq)
        dB_list_name_pesq['avg'] = 0
        dB_list_name_stoi['avg'] = 0
        dB_list_name_SISDR['avg'] = 0
        improve_pesq = 0
        for key, value in dB_list_pesq.items():
            avg_pesq=np.mean(value,0)
            pesq_list=[[avg_pesq[0],avg_pesq[1],avg_pesq[2]],"avg_pesq"]
            dB_list_name_pesq[key].append([pesq_list])
            dB_list_name_pesq['avg']+=avg_pesq[1]/num
            improve_pesq+=avg_pesq[2]/num
        for key, value in dB_list_stoi.items():
            avg_stoi=np.mean(value,0)
            stoi_list=[[avg_stoi[0],avg_stoi[1],avg_stoi[2]],"avg_stoi"]
            dB_list_name_stoi[key].append([stoi_list])
            dB_list_name_stoi['avg']+=avg_stoi[1]/num
        for key, value in dB_list_SISDR.items():
            avg_SISDR=np.mean(value,0)
            SISDR_list=[[avg_SISDR[0],avg_SISDR[1],avg_SISDR[2]],"avg_SISDR"]
            dB_list_name_SISDR[key].append([SISDR_list])
            dB_list_name_SISDR['avg']+=avg_SISDR[1]/num

            
        noise_avg=list()
        for key, value in dB_noise_pesq.items():
            avg_pesq=np.mean(value,0)
            noise_avg.append([key,np.round(avg_pesq,decimals=3)])
            # if key==dB_noise_pesq.keys[-1]:
            #     print(noise_avg)
    print(noise_avg)
    dB_list_name_pesq['avg']=round(dB_list_name_pesq['avg'],3)
    dB_list_name_stoi['avg']=round(dB_list_name_stoi['avg'],3)
    dB_list_name_SISDR['avg']=round(dB_list_name_SISDR['avg'],3)
    pesq_avg=dB_list_name_pesq['avg']
    stoi_avg=dB_list_name_stoi['avg']
    SISDR_avg=dB_list_name_SISDR['avg']
    print(f'pesq_avg:{pesq_avg} stoi_avg:{stoi_avg} improve_pesq:{round(improve_pesq,3)} SISDR:{SISDR_avg} ')
    return {'pesq' : dB_list_name_pesq ,'stoi' : dB_list_name_stoi,'SISDR' : dB_list_name_SISDR,'noise':noise_avg}
def evaluate_for_enhanced(args, dataset, model):
    dB_list_pesq=dict();dB_list_name_pesq=dict();dB_list_stoi=dict();dB_list_name_stoi=dict()
    if args.outside_test :
        for i in ['-7.5' ,'-2.5' ,'2.5' ,'7.5' ]:
            dB_list_pesq[i] = list()
            dB_list_name_pesq[i] = list()

            dB_list_stoi[i] = list()
            dB_list_name_stoi[i] = list()
            test_noise_file="outside_test/noise"
    else:
        for i in ['-10','-5' ,'0' ,'5' ,'10' ]:
            dB_list_pesq[i] = list()
            dB_list_name_pesq[i] = list()

            dB_list_stoi[i] = list()
            dB_list_name_stoi[i] = list()
            test_noise_file="test/noise"
    noise_dir=os.path.join(args.dataset_dir,test_noise_file)
    noise_file = os.listdir(noise_dir)
    dB_noise_pesq = {}
    for i in noise_file:
        dB_noise_pesq[os.path.splitext(i)[0]]=list()
    
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataset)) as pbar:
            for example in dataset:
                # Load source references in their original sr and channel number
                target_sources = utils.load(example['target'], sr=16000, mono=True)[0].flatten()
                # Predict using mixture
                pred_sources  = predict_song(args, example["input"], model).flatten()
                # write wav
                file_name=os.path.basename(example['input'])
                if args.write_to_wav:
                    utils.write_wav(os.path.join(args.output,'enhance_'+file_name), pred_sources.T, args.sr)
                fname,ext = os.path.splitext(file_name)
                text=fname.split("_",4)
                # Evaluate pesq
                enhance_pesq=pesq(target_sources, pred_sources ,16000)
                # Evaluate stoi
                enhance_stoi = stoi(target_sources, pred_sources, 16000, extended=False)

                filename=os.path.basename(example['input'])
                noise_name=filename.split("_")[0]
                dB_noise_pesq[noise_name].append([enhance_pesq])

                dB_list_pesq[text[4]].append(enhance_pesq)
                dB_list_name_pesq[text[4]].append([enhance_pesq,filename])

                dB_list_stoi[text[4]].append(enhance_stoi)
                dB_list_name_stoi[text[4]].append([enhance_stoi,filename])
                pbar.update(1)

        dB_list_name_pesq['avg'] = 0
        dB_list_name_stoi['avg'] = 0
        num=len(dB_list_pesq)
        for key, value in dB_list_pesq.items():
            avg_pesq=np.mean(value,0)
            dB_list_name_pesq[key].append([avg_pesq,"avg_pesq"])
            dB_list_name_pesq['avg']+=avg_pesq/num

        for key, value in dB_list_stoi.items():
            avg_stoi=np.mean(value,0)
            dB_list_name_stoi[key].append([avg_stoi,"avg_stoi"])
            dB_list_name_stoi['avg']+=avg_stoi/num

        noise_avg=list()
        for key, value in dB_noise_pesq.items():
            avg_pesq=np.mean(value,0)
            noise_avg.append([key,avg_pesq])

    print(noise_avg)
    pesq_avg=dB_list_name_pesq['avg']
    stoi_avg=dB_list_name_stoi['avg']
    print(f'pesq_avg:{pesq_avg} stoi_avg:{stoi_avg} ')
    return {'pesq' : dB_list_name_pesq ,'stoi' : dB_list_name_stoi,'noise':noise_avg}
def ling_evaluate(args, dataset, model):
    model.eval()
    length = len(dataset)
    print(length)
    all_1=np.zeros([length])
    all_2=np.zeros([length])
    i=0
    with torch.no_grad():
        with tqdm(total=len(dataset)) as pbar:
            for example in dataset:
                # Load source references in their original sr and channel number
                target_sources = utils.load(example['target'], sr=16000, mono=True)[0].flatten()
                #input_sources = utils.load(example['input'], sr=16000, mono=True)[0].flatten()

                # Predict using mixture
                pred_sources  = predict_song(args, example["input"], model).flatten()
                # Evaluate pesq
                #input_pesq=pysepm.pesq(target_sources, input_sources,16000)[1]
                enhance_pesq=pysepm.pesq(target_sources, pred_sources ,16000)[1]
                all_1[i]=enhance_pesq
                #all_2[i]=enhance_pesq
                # Evaluate stoi
                # input_stoi = stoi(target_sources, input_sources, 16000, extended=False)
                # enhance_stoi = stoi(target_sources, pred_sources, 16000, extended=False)
                # print(f'input_pesq:{input_pesq} enhance_pesq:{enhance_pesq} improve_pesq:{enhance_pesq-input_pesq} ')
                pbar.update(1)
                i+=1
    print(np.mean(all_1),np.mean(all_2))
    return 
def validate(args, model, criterion, test_data):
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
    matrics = { 'PESQ' : [np.zeros(len_data+1),np.zeros(len_data+1),np.zeros(len_data+1)] ,
                'STOI' : [np.zeros(len_data+1),np.zeros(len_data+1),np.zeros(len_data+1)] ,
                'SISDR' : [np.zeros(len_data+1),np.zeros(len_data+1),np.zeros(len_data+1)]  }

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
                matrics['STOI'][0][example_num]=values1
                matrics['STOI'][1][example_num]=values2
                matrics['STOI'][2][example_num]=values2 - values1

            # pesq
            values1=round(pesq(target, inputs,16000),2)
            values2=round(pesq(target, pred ,16000),2)
            if(~np.isnan(values1) and  ~np.isnan(values2) ):
                matrics['PESQ'][0][example_num]=values1
                matrics['PESQ'][1][example_num]=values2
                matrics['PESQ'][2][example_num]=values2 - values1
            # SISDR
            enhance_data=nussl.AudioSignal(audio_data_array=pred,sample_rate=16000)
            input_data=nussl.AudioSignal(audio_data_array=inputs,sample_rate=16000)
            target_data=nussl.AudioSignal(audio_data_array=target,sample_rate=16000)
            evaluator=nussl.evaluation.BSSEvalScale(target_data  ,input_data )
            scores = evaluator.evaluate()
            values1=scores['source_0']['SI-SDR'][0]
            evaluator=nussl.evaluation.BSSEvalScale(target_data  ,enhance_data )
            scores = evaluator.evaluate()
            values2=scores['source_0']['SI-SDR'][0]
            if(~np.isnan(values1) and  ~np.isnan(values2) ):
                matrics['SISDR'][0][example_num]=values1
                matrics['SISDR'][1][example_num]=values2
                matrics['SISDR'][2][example_num]=values2 - values1
            
            pbar.set_description("Current loss: {:.4f}".format(total_loss))
            pbar.update(1)
    val_improve_pesq=np.nanmean(matrics['PESQ'][2])
    val_improve_stoi=np.nanmean(matrics['STOI'][2])
    val_improve_SISDR=np.nanmean(matrics['SISDR'][2])
    val_enhance_pesq=np.nanmean(matrics['PESQ'][1])
    val_enhance_stoi=np.nanmean(matrics['STOI'][1])
    val_enhance_SISDR=np.nanmean(matrics['SISDR'][1])
    val_input_pesq=np.nanmean(matrics['PESQ'][0])
    val_input_STOI=np.nanmean(matrics['STOI'][0])
    val_input_SISDR=np.nanmean(matrics['SISDR'][0])
    print(f'val_input_pesq={val_input_pesq}')
    print(f'val_improve_pesq={val_improve_pesq}')

    print(f'val_input_STOI={val_input_STOI}')
    print(f'val_improve_stoi={val_improve_stoi}')

    print(f'val_input_SISDR={val_input_SISDR}')
    print(f'val_improve_SISDR={val_improve_SISDR}')

    # writer.add_scalar("val_enhance_pesq", np.nanmean(avg_enhance_pesq), state["epochs"])
    # writer.add_scalar("val_improve_pesq", np.nanmean(avg_improve_pesq), state["epochs"])

    # writer.add_scalar("val_enhance_stoi", np.nanmean(avg_enhance_stoi), state["epochs"])
    # writer.add_scalar("val_improve_stoi", np.nanmean(avg_improve_stoi), state["epochs"])
    
    return total_loss,[val_enhance_pesq,val_improve_pesq,val_enhance_stoi,val_improve_stoi,val_enhance_SISDR,val_improve_SISDR]
def validate_ori(args, model, criterion, test_data):
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)
    # pesq_test=(random.randrange(len(test_data)))
    # VALIDATE
    model.eval()
    
    # print('np.zeros(len(test_data) // args.batch_size)',np.zeros(len(test_data) // args.batch_size))

    total_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, targets) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                targets = targets.cuda()

            outputs, avg_loss = compute_loss(model, x, targets, criterion)
            total_loss += (1. / float(example_num + 1)) * (avg_loss - total_loss)
            
            
    return total_loss