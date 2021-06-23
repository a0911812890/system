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
from Loss import customLoss,RL_customLoss
from pypesq import pesq
import oneMask_criterion
def compute_output(model, inputs):
    all_outputs = model(inputs)

    return all_outputs


### MSE ###
def KD_compute_loss(model,teacher_model, inputs, targets, criterion,alpha, compute_grad=False,KD_method='A'):
    student_all_outputs = model(inputs)
    teacher_all_outputs = teacher_model(inputs).detach()

    if KD_method =='A':
        student_loss = criterion(student_all_outputs,targets,1-alpha) 
        KD_loss = criterion(student_all_outputs, teacher_all_outputs,alpha)
    elif  KD_method =='B':
        student_loss = criterion(student_all_outputs,targets,1) 
        KD_loss = criterion(student_all_outputs, teacher_all_outputs,alpha)
    elif  KD_method =='C':
        student_loss = criterion(student_all_outputs,targets,alpha) 
        KD_loss = criterion(student_all_outputs, teacher_all_outputs,1)
    else :
        raise Exception(f"unknown KD_method{KD_method} !!! (A or B)")

    total_loss = student_loss + KD_loss

    if compute_grad:
        total_loss.backward()

    student_avg_loss = student_loss.item() / float(len(student_all_outputs))
    total_avg_loss = total_loss.item() / float(len(student_all_outputs))
    KD_avg_loss = KD_loss.item() / float(len(student_all_outputs))
    return student_all_outputs, student_avg_loss  ,total_avg_loss ,KD_avg_loss


def loss_for_sample(model, inputs, targets):
    loss = 0
    with torch.no_grad():
        all_outputs = model(inputs).detach()
        loss = torch.mean(torch.pow((all_outputs-targets),2),2)/len(all_outputs)
    return loss


def compute_loss(model, inputs, targets, criterion, compute_grad=False):
    all_outputs = model(inputs)
    loss = criterion(all_outputs, targets)
    if compute_grad:
        loss.backward()

    avg_loss = loss.item() / float(len(all_outputs))
    return all_outputs, avg_loss 

def range_method(memory_iter,R):
    SORT, indices = torch.sort(memory_iter,0)
    bo=normalize(R,0.2,-0.2)*32 #-0.5 # 0.25
    # print('type',type(bo))
    bo=int(bo)
    label=31-bo
    # print(f'R ={R} bo={bo} label={label}')
    if label>=len(memory_iter):
        label=len(memory_iter)-1
    elif label<0:
        label=0
    bound=SORT[label]
    norm_r = (memory_iter - bound) / (torch.max(memory_iter) - torch.min(memory_iter))
    norm_r /= 32 * 691
    norm_r = norm_r.cuda()
    return norm_r

def value_method(memory_iter,R):
    bound=(torch.max(memory_iter) - torch.min(memory_iter))*(1-normalize(R,0.1,-0.1))
    bound+=torch.min(memory_iter)
    norm_r = (memory_iter - bound) / (torch.max(memory_iter) - torch.min(memory_iter))
    norm_r /= 32 * 691
    norm_r = norm_r.cuda()
    return norm_r

def normalize(inputs,MAX,MIN):
    
    output = (inputs-MIN)/(MAX-MIN)
    if output>1:
        output=(MAX-MIN)/(MAX-MIN)
        #print('normalize out range to 1')
    elif output<0:
        output=(MIN-MIN)/(MAX-MIN)
        print('normalize out range to 0')
    return output



### SISNR ###

def sisnr_KD_compute_loss(model,teacher_model, inputs, targets, criterion,alpha,batch_size, compute_grad=False):
    student_all_outputs = model(inputs)
    teacher_all_outputs = teacher_model(inputs).detach()

    ori_Loss, ori_max_snr, ori_estimate_source, ori_reorder_estimate_source = oneMask_criterion.cal_loss(targets, all_outputs, 
                                                                                                        torch.tensor(batch_size*[28793]).cuda(),1)

    KD_Loss, KD_max_snr, KD_estimate_source, KD_reorder_estimate_source = oneMask_criterion.cal_loss(targets, all_outputs,
                                                                                                     torch.tensor(batch_size*[28793]).cuda(),alpha)

    Loss =  KD_Loss + ori_Loss
    if compute_grad:
        Loss.backward()

    avg_ori_sisnr=torch.mean(ori_max_snr).item()
    avg_KD_sisnr=torch.mean(KD_max_snr).item()
    avg_sisnr=avg_ori_sisnr+avg_KD_sisnr
    print(f'avg_ori_sisnr={avg_ori_sisnr}')
    print(f'avg_KD_sisnr={avg_KD_sisnr}')
    print(f'avg_sisnr={avg_sisnr}')
    

    if compute_grad:
        Loss.backward()

    return student_all_outputs,avg_ori_sisnr,avg_sisnr


def sisnr_loss_for_sample(model, inputs, targets,batch_size):
    loss = 0
    with torch.no_grad():
        all_outputs = model(inputs).detach()
        Loss, max_snr, estimate_source, reorder_estimate_source = oneMask_criterion.cal_loss(targets, all_outputs, torch.tensor(batch_size*[28793]).cuda())
    return max_snr

def sisnr_compute(model, inputs, targets,batch_size, compute_grad=False):
    loss = 0
    all_outputs = model(inputs)
    Loss, max_snr, estimate_source, reorder_estimate_source = oneMask_criterion.cal_loss(targets, all_outputs, torch.tensor(batch_size*[28793]).cuda())
    if compute_grad:
        Loss.backward()
    avg_sisnr=torch.mean(max_snr).item()
    print(avg_sisnr)
    return all_outputs, avg_sisnr


### RL ###    
def RL_compute_loss(RL_alpha, reward,criterion):
    label = torch.zeros(len(RL_alpha),1)
    label = label.cuda()

    for i in range(len(label)):
        label[i]=RL_alpha[i].detach()+reward[i] 
        if label[i]>1:
            # print(f'label bigger than 1 ,modify to 1')
            label[i]=1
        elif label[i]<0:
            # print(f'label smaller than 0 ,modify to 0')
            label[i]=0
    loss = criterion(RL_alpha, label)
    loss.backward()
    return loss

def smoothing(rewards):
    for i in range(len(rewards)):
        if (rewards[i]>0):
            rewards[i]=torch.sqrt(rewards[i])
        elif (rewards[i]<0):
            rewards[i]=-torch.sqrt(-rewards[i])
    return rewards

def get_rewards(backward_KD_loss,backward_copy_loss,backward_copy2_loss,student_KD_loss,len_data,decay):
    improvement=torch.zeros([len(backward_KD_loss),1]).cuda() 
    same=0
    for i in range(len(backward_KD_loss)):
        if((backward_KD_loss[i] < backward_copy_loss[i]) and(backward_KD_loss[i] < backward_copy2_loss[i]) ):
            improvement[i]=backward_KD_loss[i] - backward_KD_loss[i]
        elif((backward_copy_loss[i] < backward_KD_loss[i]) and (backward_copy_loss[i] < backward_copy2_loss[i])):
            improvement[i]=backward_KD_loss[i] - backward_copy_loss[i]
        elif((backward_copy2_loss[i] < backward_KD_loss[i]) and (backward_copy2_loss[i] < backward_copy_loss[i])):
            improvement[i]=backward_copy2_loss[i] - backward_KD_loss[i] 
        if((backward_copy_loss[i] < backward_KD_loss[i]) and(backward_KD_loss[i] < backward_copy2_loss[i])):
            same+=1/len_data
        elif ((backward_copy_loss[i] > backward_KD_loss[i]) and(backward_KD_loss[i] > backward_copy2_loss[i])):
            same+=1/len_data

    rewards = (improvement/student_KD_loss).detach()
    rewards = smoothing(rewards)
    # calculating before_decay
    before_decay=torch.mean(rewards)

    rewards = rewards/decay
    

    return rewards,same,before_decay


###   data process ###
def mkdir_and_get_path(args):
    if args.test == True:
        model_name = "test"
    else:
        model_name = args.model_name
    log_dir = os.path.join(args.model_base_path,model_name,'logs')
    checkpoint_dir = os.path.join(args.model_base_path,model_name,'checkpoints')
    result_dir = os.path.join(args.model_base_path,model_name,'results')
    if not os.path.isdir(log_dir):
        os.makedirs( log_dir )
    if not os.path.isdir(checkpoint_dir):
        os.makedirs( checkpoint_dir )
    if not os.path.isdir(result_dir):
        os.makedirs( result_dir )
    if not os.path.isdir(result_dir):
        os.makedirs( result_dir )
    return log_dir,checkpoint_dir,result_dir


def save_result(data,dir_path,name):
    
    with open(os.path.join(dir_path,name+ "_results.pkl"), "wb") as f:
        pickle.dump(data, f)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(dir_path,name+ "_results.csv"),sep=',')
   
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