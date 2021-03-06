import os
import pandas as pd
import soundfile
import torch
import numpy as np
import librosa
from datetime import date,datetime
import pickle
import torch
import torch.nn as nn
from pypesq import pesq
def compute_output(model, inputs):
    '''
    Computes outputs of model with given inputs. Does NOT allow propagating gradients! See compute_loss for training.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''

    all_outputs = model(inputs)

    return all_outputs

def KD_compute_loss(model,teacher_model, inputs, targets, criterion,alpha=0.5, compute_grad=False):
    
    student_loss = 0
    KD_loss = 0
    
    student_all_outputs = model(inputs)
    teacher_all_outputs = teacher_model(inputs)

    # input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet    
    # target=torch.mean(targets[0], 0).cpu().numpy()
    # pred1=torch.mean(student_all_outputs[0], 0).detach().cpu().numpy()
    # pred2=torch.mean(teacher_all_outputs[0], 0).detach().cpu().numpy()
    # inputs=input_centre.cpu().numpy()

    # values1=round(pesq(target,inputs,16000),5)
    # values2=round(pesq(target,pred1 ,16000),5)
    # values3=round(pesq(target,pred2 ,16000),5)

    student_loss = criterion(student_all_outputs, targets)
    KD_loss = criterion(student_all_outputs, teacher_all_outputs)
    total_loss = (1-alpha)*student_loss + alpha*KD_loss

    if compute_grad:
        total_loss.backward()

    student_avg_loss = student_loss.item() / float(len(student_all_outputs))
    total_avg_loss = total_loss.item() / float(len(student_all_outputs))
    #KD_avg_loss = KD_loss.item() / float(len(student_all_outputs))
    return student_all_outputs, student_avg_loss  ,total_avg_loss 




def RL_compute_loss(RL_alpha_array, reward,criterion):
    sign=0
    # value=(1e-6)/diff
    label = torch.zeros(len(RL_alpha_array),1)
    for i in range(len(label)):
        if reward>0:
            label[i][0]=1
            sign=1
        else:
            label[i][0]=0
            sign=-1
    
    label = label.cuda()
    loss = criterion(RL_alpha_array, label)

    reward=np.abs(reward)
    reward=KD_normalize(reward)
    #print(f'reward={reward}')
  
    loss = loss*reward
    loss.backward()
    
    sign_normalize_reward=sign*reward
    return loss,sign_normalize_reward

def compute_loss(model, inputs, targets, criterion, compute_grad=False):
    '''
    Computes gradients of model with given inputs and targets and loss function.
    Optionally backpropagates to compute gradients for weights.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param inputs: Input mixture
    :param targets: Target sources
    :param criterion: Loss function to use (L1, L2, ..)
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    loss = 0
    all_outputs = model(inputs)
    loss += criterion(all_outputs, targets)
    if compute_grad:
        loss.backward()

    avg_loss = loss.item() / float(len(all_outputs))

    return all_outputs, avg_loss

def KD_normalize(inputs,MAX=8*10e-5 , MIN=0):
    if inputs<10e-9:
        inputs=10e-9
    return (inputs-MIN)/(MAX-MIN)


def save_result(data,dir_path,name):
    
    with open(os.path.join(dir_path,name+ "_results.pkl"), "wb") as f:
        pickle.dump(data, f)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(dir_path,name+ "_results.csv"),sep=',')

def args_to_csv(args,dir_path=""):
    
    arg = list()
    if(dir_path==""):
        dir_path=args.log_dir
    x=vars(args)
    for index,data in enumerate(x):
        arg.append([data,x[data]])
    arg=pd.DataFrame(arg)
    arg.to_csv(os.path.join(dir_path,"args.csv"))
def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr

def write_wav(path, audio, sr):
    soundfile.write(path, audio, sr, "PCM_16")

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def set_cyclic_lr(optimizer, it, epoch_it, cycles, min_lr, max_lr):
    cycle_length = epoch_it // cycles
    curr_cycle = min(it // cycle_length, cycles-1)
    curr_it = it - cycle_length * curr_cycle

    new_lr = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos((float(curr_it) / float(cycle_length)) * np.pi))
    set_lr(optimizer, new_lr)

def resample(audio, orig_sr, new_sr, mode="numpy"):
    if orig_sr == new_sr:
        return audio

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    out = librosa.resample(audio, orig_sr, new_sr, res_type='kaiser_fast')

    if mode == "pytorch":
        out = torch.tensor(out)
    return out

class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
    }, path)

def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoitns only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state

def load_latest_model_from(model, optimizer, location, cuda):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)
    return load_model(model, optimizer, newest_file, cuda)