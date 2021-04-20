import argparse
import os
from functools import partial
import pandas as pd
import numpy as np
import time
import copy
from datetime import date,datetime
from tqdm import tqdm
from pypesq import pesq

#
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam,SGD
#
import utils 
from data import  SeparationDataset,  crop ,get_folds
from test import evaluate, validate
from waveunet import Waveunet
from RL import RL
from Memory import Memory
from Loss import customLoss,RL_customLoss
#
    

def main(args):
    # set seed
    torch.cuda.manual_seed_all(1)
    np.random.seed(0)

    # filter array
    num_features = [args.features*i for i in range(1, args.levels+2+args.levels_without_sample)] 
    teacher_num_features = [24*i for i in range(1, args.levels+2+args.levels_without_sample)] 

    # 確定 輸出大小
    target_outputs = int(args.output_size * args.sr)
    # 訓練才保存模型設定參數
    if args.test is False:
        utils.args_to_csv(args)


    # 設定teacher and student and student_for_backward 超參數

    student_KD = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
    if args.cuda:
        student_KD = utils.DataParallel(student_KD)
        print("move student_KD to gpu\n")
        student_KD.cuda()
    student_size = sum(p.numel() for p in student_KD.parameters())
    print('student_KD: ', student_KD.shapes)
    print('student_parameter count: ', str(student_size))

    if args.teacher_model is not None:
        teacher_model = Waveunet(args.channels, teacher_num_features, args.channels,levels=args.levels, 
                encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                conv_type=args.conv_type, res=args.res)

        student_copy = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)

        policy_network=RL(n_inputs=2,kernel_size=6,stride=1,conv_type=args.conv_type,pool_size=4)
        if args.cuda:
            teacher_model = utils.DataParallel(teacher_model)
            policy_network = utils.DataParallel(policy_network)
            student_copy = utils.DataParallel(student_copy)
            
            print("move teacher to gpu\n")
            teacher_model.cuda()
            print("student_copy  to gpu\n")
            student_copy.cuda()
            print("move policy_network to gpu\n")
            policy_network.cuda()
        teacher_size=sum(p.numel() for p in teacher_model.parameters())
        print('teacher_model_parameter count: ', str(teacher_size))
        print('RL_parameter count: ', str(sum(p.numel() for p in policy_network.parameters())))
        print(f'compression raito :{100*(student_size/teacher_size)}%')
    

    
    # print(model)
    if args.test is False:
        writer = SummaryWriter(args.log_dir)

    # ### DATASET

    dataset = get_folds(args.dataset_dir,args.outside_test)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop, shapes=student_KD.shapes)
    # Data augmentation function for training
    if args.test is False:
        train_data = SeparationDataset(dataset, "train", args.sr, args.channels, student_KD.shapes, False, args.hdf_dir, audio_transform=crop_func)
        val_data = SeparationDataset(dataset, "test", args.sr, args.channels, student_KD.shapes, False, args.hdf_dir, audio_transform=crop_func)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn,pin_memory=True)
    test_data = SeparationDataset(dataset, "test", args.sr, args.channels, student_KD.shapes, False, args.hdf_dir, audio_transform=crop_func)
    
        

    # ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")
    My_criterion = customLoss()
    # Set up optimizer
    KD_optimizer = Adam(params=student_KD.parameters(), lr=args.lr)
    if args.teacher_model is not None:
        copy_optimizer = Adam(params=student_copy.parameters(), lr=args.lr)
        PG_optimizer = Adam(params=policy_network.parameters(), lr=args.RL_lr)
    
    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_pesq" : -np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_RL_model is not None:
        print("Continuing full RL_model from checkpoint " + str(args.load_RL_model))
        policy_network.load_state_dict(torch.load(args.load_RL_model))
    if args.load_model is not None:
        print("Continuing full student_KD from checkpoint " + str(args.load_model))
        if args.test is False and args.teacher_model is not None :
            state = utils.load_model(student_copy, copy_optimizer, args.load_model, args.cuda)
        state = utils.load_model(student_KD, KD_optimizer, args.load_model, args.cuda)

    # load teacher model
    if args.teacher_model is not None :
        print("load teacher model" + str(args.teacher_model))
        teacher_state = utils.load_model(teacher_model, None, args.teacher_model, args.cuda)
        teacher_model.eval()

    if args.test is False:
        print('TRAINING START')

        batch_num=(len(train_data) // args.batch_size)

        KD_to_copy=True
        print(f'KD_to_copy={KD_to_copy}')
        # initialized weight


        # store_KD_rate
        remain=20

        while state["epochs"] < 150:
            if args.teacher_model is not None:
                if KD_to_copy is True:
                    student_copy.load_state_dict(copy.deepcopy(student_KD.state_dict()))
                    copy_optimizer.load_state_dict(copy.deepcopy(KD_optimizer.state_dict()))
                else:
                    student_KD.load_state_dict(copy.deepcopy(student_copy.state_dict()))
                    KD_optimizer.load_state_dict(copy.deepcopy(copy_optimizer.state_dict()))

                student_copy.train()
            memory_bank=[]
            print("epoch:"+str(state["epochs"]))
            student_KD.train()


            # monitor_value    
            total_r=0
            avg_origin_loss=0
            all_avg_KD_rate=0
            with tqdm(total=len(dataloader)) as pbar:
                for example_num, (x, targets) in enumerate(dataloader):
                    if args.cuda:
                        x = x.cuda()
                        targets = targets.cuda()
                    
                    if args.teacher_model is not None:
                        
                        # Set LR for this iteration  
                        utils.set_cyclic_lr(KD_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        utils.set_cyclic_lr(copy_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)

                        # forward student and teacher  get output
                        student_KD_output, student_KD_loss=utils.compute_loss(student_KD, x, targets, criterion,compute_grad=False)
                        teacher_output, _=utils.compute_loss(teacher_model, x, targets, criterion,compute_grad=False)

                        # PG_state
                        PG_state=torch.cat((targets-student_KD_output,teacher_output-student_KD_output),1)

                        # forward RL get alpha
                        alpha=policy_network(PG_state)
                        nograd_alpha=alpha.detach()
                        
                        avg_KD_rate=torch.mean(nograd_alpha).item()
                        all_avg_KD_rate+=avg_KD_rate / batch_num
                        KD_optimizer.zero_grad()
                        KD_outputs, KD_hard_loss ,KD_loss ,KD_soft_loss = utils.KD_compute_loss(student_KD,teacher_model, x, targets, My_criterion,alpha=nograd_alpha,compute_grad=True)
                        KD_optimizer.step()

                        copy_optimizer.zero_grad()
                        _, _,_ ,_ = utils.KD_compute_loss(student_copy,teacher_model, x, targets, My_criterion,alpha=0,compute_grad=True)
                        copy_optimizer.step()

                        # calculate backwarded model MSE
                        backward_KD_loss = utils.loss_for_sample(student_KD, x, targets)
                        

                        # calculate r
                        r = (student_KD_loss - backward_KD_loss).detach()
                        r /= student_KD_loss
                        avg_origin_loss += student_KD_loss / batch_num
                        
                        # backward RL 
                        PG_optimizer.zero_grad()
                        PG_loss=utils.RL_compute_loss(alpha,r,nn.MSELoss())
                        PG_optimizer.step()

                        # avg_r
                        avg_r=torch.mean(r)
                        total_r+=avg_r.item()
                        print(f'improve ratio = {avg_r}')
                        # append to memory bank
                        memory_bank.append(Memory(PG_state,nograd_alpha,r))
                        # print('memory_bank')
                        # print(memory_bank[-1].alpha)
                        # print info
                        print(f'avg_KD_rate = {avg_KD_rate} ')
                        print(f'student_KD_loss             = {student_KD_loss}')
                        print(f'backward_student_KD_loss    = {np.mean(backward_KD_loss.cpu().detach().numpy())}')
                        # print(f'student_copy_loss           = {np.mean(backward_copy_loss.cpu().detach().numpy())}')
                        print(f'avg_r                       = {avg_r}')
                        print(f'total_r                     = {total_r}')
                        print(f'PG_loss                     = {PG_loss}')
                        # add to tensorboard
                        
                        writer.add_scalar("student_KD_loss", student_KD_loss, state["step"])
                        writer.add_scalar("backward_student_KD_loss", np.mean(backward_KD_loss.cpu().detach().numpy()), state["step"])
                        writer.add_scalar("KD_loss", KD_loss, state["step"])
                        writer.add_scalar("KD_hard_loss", KD_hard_loss, state["step"])
                        writer.add_scalar("KD_soft_loss", KD_soft_loss, state["step"])
                        writer.add_scalar("avg_KD_rate",avg_KD_rate, state["step"])
                        writer.add_scalar("PG_loss",PG_loss, state["step"])
                        writer.add_scalar("r", avg_r, state["step"])

                    else: # no KD training
                        utils.set_cyclic_lr(KD_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        KD_optimizer.zero_grad()
                        KD_outputs, KD_hard_loss = utils.compute_loss(student_KD, x, targets, nn.MSELoss(), compute_grad=True)
                        KD_optimizer.step()
                        avg_origin_loss+=KD_hard_loss/batch_num
                        writer.add_scalar("student_KD_loss", KD_hard_loss, state["step"])
                   

                    
                    ### save wav ####
                    if example_num % args.example_freq == 0:
                        input_centre = torch.mean(x[0, :, student_KD.shapes["output_start_frame"]:student_KD.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                        
                        target=torch.mean(targets[0], 0).cpu().numpy()
                        pred=torch.mean(KD_outputs[0], 0).detach().cpu().numpy()
                        inputs=input_centre.cpu().numpy()


                        writer.add_audio("input:", input_centre, state["step"], sample_rate=args.sr)
                        writer.add_audio("pred:", torch.mean(KD_outputs[0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio("target", torch.mean(targets[0], 0), state["step"], sample_rate=args.sr)

                    state["step"] += 1
                    pbar.update(1)
            all_avg_KD_rate
            # VALIDATE
            val_loss,val_metrics = validate(args, student_KD, criterion, val_data)
            print("ori VALIDATION FINISHED: LOSS: " + str(val_loss))
            
            
            writer.add_scalar("avg_origin_loss", avg_origin_loss, state["epochs"])
            choose_val=0
            
            if args.teacher_model is not None :
                val_loss_copy,val_metrics_copy = validate(args, student_copy, criterion, val_data)
                print("copy VALIDATION FINISHED: LOSS: " + str(val_loss_copy))
                
                if val_metrics[0]>val_metrics_copy[0]:
                    KD_to_copy=True
                    choose_val=val_metrics
                else:
                    KD_to_copy=False
                    choose_val=val_metrics_copy


                if KD_to_copy is True:
                    print('RL choose is better ')
                else:
                    print('copy is better')

                for i in range(len(nograd_alpha)):
                    writer.add_scalar("KD_rate_"+str(i), nograd_alpha[i], state["epochs"])

                writer.add_scalar("all_avg_KD_rate", all_avg_KD_rate, state["epochs"])
                writer.add_scalar("val_loss_copy", val_loss_copy, state["epochs"])
                writer.add_scalar("total_r", total_r, state["epochs"])
                writer.add_scalar("avg_origin_loss", avg_origin_loss, state["epochs"])

                RL_checkpoint_path = os.path.join(args.checkpoint_dir, "RL_checkpoint_" + str(state["epochs"]))
                utils.save_model(policy_network, PG_optimizer, state, RL_checkpoint_path)
            else:
                choose_val=val_metrics
            
            writer.add_scalar("val_enhance_pesq",choose_val[0], state["epochs"])
            writer.add_scalar("val_improve_pesq",choose_val[1], state["epochs"])
            writer.add_scalar("val_enhance_stoi",choose_val[2], state["epochs"])
            writer.add_scalar("val_improve_stoi",choose_val[3], state["epochs"])
            writer.add_scalar("val_loss", val_loss, state["epochs"])

            # EARLY STOPPING CHECK
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["epochs"]))
            if choose_val[0] < state["best_pesq"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_pesq"] = choose_val[0]
                state["best_checkpoint"] = checkpoint_path

            # CHECKPOINT
            print("Saving model...")
            utils.save_model(student_KD, KD_optimizer, state, checkpoint_path)
            state["epochs"] += 1
    if args.test is False:
        writer.close()


    #### TESTING ####
    # Test loss
    print("TESTING")
    # eval metrics
    test_metrics = evaluate(args, dataset["test"], student_KD)
    test_pesq=test_metrics['pesq']
    test_stoi=test_metrics['stoi']
    test_noise=test_metrics['noise']

    if args.test is False:
        info=args.model_name
        path=os.path.join(args.result_dir,info)
    else:
        PATH=args.load_model.split("/")
        info=PATH[-3]+"_"+PATH[-1]
        if(args.outside_test==True):
            info+="_outside_test"
        print(info)
        path=os.path.join(args.result_dir,info)
    if not os.path.exists(path):
        os.makedirs(path)
    utils.save_result(test_pesq,path,"pesq")
    utils.save_result(test_stoi,path,"stoi")
    utils.save_result(test_noise,path,"noise")
    #### TESTING ####
    



if __name__ == '__main__':
    ## TRAIN PARAMETERS
    model_base_path='/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/model'
    model_name = "self_reward"
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--outside_test', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--test', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loader worker threads (default: 4)')
    parser.add_argument('--features', type=int, default=12,
                        help='Number of feature channels per layer')
    parser.add_argument('--test_dir', type=str, default="/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/results",
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/media/hd03/sutsaiwei_data/data/mydata",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/hdf/snr_hdf",
                        help='Dataset path')

    parser.add_argument('--teacher_model', type=str, default=None,
                        help='load a  pre-trained teacher model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--load_RL_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--RL_lr', type=float, default=5e-5,
                        help='Initial RL_learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=7,
                        help="Number of DS/US blocks ")
    parser.add_argument('--levels_without_sample', type=int, default=5,
                        help="Number of non-updown-DS/US blocks ") 
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--encoder_kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")      
    parser.add_argument('--decoder_kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")                                 
    parser.add_argument('--output_size', type=float, default=1.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=2,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="bn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--feature_growth', type=str, default="add",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--output', type=str, default="/media/hd03/sutsaiwei_data/data/yunwen_data/test/enhance", help="Output path (same folder as input path if not set)")
    args = parser.parse_args()

    if args.test==True:
            model_name = "test"
    parser.add_argument('--log_dir', type=str, default=os.path.join(model_base_path,model_name,'logs'),
                        help='Folder to write logs into')
    parser.add_argument('--result_dir', type=str, default=os.path.join(model_base_path,model_name,'results'),
                        help='Folder to write results into')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(model_base_path,model_name,'checkpoints'),
                        help='Folder to write checkpoints into')
    parser.add_argument('--model_name', type=str, default=model_name,
                        help='model_name')
    args = parser.parse_args()
    if not os.path.isdir(args.log_dir):
        os.makedirs( args.log_dir )
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs( args.checkpoint_dir )
    if not os.path.isdir(args.result_dir):
        os.makedirs( args.result_dir )
    main(args)
