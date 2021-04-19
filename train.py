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
from Loss import customLoss,RL_customLoss
#

def main(args):
    # filter array
    torch.cuda.manual_seed_all(1)
    num_features = [args.features*i for i in range(1, args.levels+2+args.levels_without_sample)] 
    teacher_num_features = [24*i for i in range(1, args.levels+2+args.levels_without_sample)] 
    # 確定 輸出大小
    target_outputs = int(args.output_size * args.sr)
    # 訓練才保存模型設定參數
    if args.test is False:
        utils.args_to_csv(args)


    # 設定teacher and student and student_for_backward 超參數

    model = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
    if args.cuda:
        model = utils.DataParallel(model)
        print("move model to gpu\n")
        model.cuda()
    

    if args.teacher_model is not None:
        teacher_model = Waveunet(args.channels, teacher_num_features, args.channels,levels=args.levels, 
                encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                conv_type=args.conv_type, res=args.res)

        model_for_backward = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)

        rl=RL(n_inputs=2,kernel_size=6,stride=1,conv_type=args.conv_type,pool_size=4)
        if args.cuda:
            teacher_model = utils.DataParallel(teacher_model)
            rl = utils.DataParallel(rl)
            model_for_backward = utils.DataParallel(model_for_backward)
            
            print("move teacher_model to gpu\n")
            teacher_model.cuda()
            print("model_for_backward model to gpu\n")
            model_for_backward.cuda()
            print("move rl to gpu\n")
            rl.cuda()

        print('teacher_model_parameter count: ', str(sum(p.numel() for p in teacher_model.parameters())))
        print('RL_parameter count: ', str(sum(p.numel() for p in rl.parameters())))
   
    print('model: ', model.shapes)
    print('student_parameter count: ', str(sum(p.numel() for p in model.parameters())))

    # print(model)
    if args.test is False:
        writer = SummaryWriter(args.log_dir)

    # ### DATASET

    dataset = get_folds(args.dataset_dir,args.outside_test)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop, shapes=model.shapes)
    # Data augmentation function for training
    if args.test is False:
        train_data = SeparationDataset(dataset, "train", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
        val_data = SeparationDataset(dataset, "test", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn,pin_memory=True)
    test_data = SeparationDataset(dataset, "test", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
    
        

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
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    optimizer_model_for_backward= 0
    if args.teacher_model is not None:
        optimizer_model_for_backward = Adam(params=model_for_backward.parameters(), lr=args.lr)
        optimizer_rl = Adam(params=rl.parameters(), lr=args.RL_lr)
    
    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_pesq" : -np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_RL_model is not None:
        print("Continuing full RL_model from checkpoint " + str(args.load_RL_model))
        rl.load_state_dict(torch.load(args.load_RL_model))
    if args.load_model is not None:
        print("Continuing full model from checkpoint " + str(args.load_model))
        if args.test is False and args.teacher_model is not None :
            state = utils.load_model(model_for_backward, optimizer_model_for_backward, args.load_model, args.cuda)
        state = utils.load_model(model, optimizer, args.load_model, args.cuda)

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
        alpha_memory=np.linspace(0,0,batch_num*args.batch_size).reshape(batch_num,args.batch_size,1)
        alpha_memory_final=np.linspace(0,0,remain).reshape(remain,1)

        alpha_memory=torch.from_numpy(alpha_memory).cuda()
        alpha_memory_final=torch.from_numpy(alpha_memory_final).cuda()

        while state["epochs"] < 150:
            if args.teacher_model is not None:
                if KD_to_copy is True:
                    model_for_backward.load_state_dict(copy.deepcopy(model.state_dict()))
                    optimizer_model_for_backward.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                else:
                    model.load_state_dict(copy.deepcopy(model_for_backward.state_dict()))
                    optimizer.load_state_dict(copy.deepcopy(optimizer_model_for_backward.state_dict()))

                model_for_backward.train()

            print("epoch:"+str(state["epochs"]))
            model.train()


            # monitor_value    
            total_RL_reward=0
            avg_origin_loss=0
            
            with tqdm(total=len(dataloader)) as pbar:
                np.random.seed()
                for example_num, (x, targets) in enumerate(dataloader):
                    if args.cuda:
                        x = x.cuda()
                        targets = targets.cuda()
                    t = time.time()
                    len_batch=len(targets)
                    # Set LR for this iteration
                    
                    # Compute loss for model    
                    if args.teacher_model is not None:
                        # set lr
                        utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        utils.set_cyclic_lr(optimizer_model_for_backward, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)

                        # forward student and teacher  get output
                        student_output, sisnr_loss=utils.sisnr_compute_loss(model, x, targets,len_batch,compute_grad=False)
                        teacher_output, _=utils.sisnr_compute_loss(teacher_model, x, targets,len_batch,compute_grad=False)
                        # concat s_out,t_out ,target
                        rl_input=torch.cat((targets-student_output,teacher_output-student_output),1)

                        # forward RL get RL_alpha
                        RL_alpha=rl(rl_input)
                        KD_rate=RL_alpha.detach()
                        
                        # save alpha to memory 
                        if example_num<len(dataloader)-1:
                            ori_KD_rate=alpha_memory[example_num]
                        else :
                            ori_KD_rate=alpha_memory_final
                        avg_ori_KD_rate=torch.mean(ori_KD_rate).item()
                        avg_KD_rate=torch.mean(KD_rate).item()
                        # student_all_outputs,avg_ori_sisnr,avg_sisnr
                        optimizer.zero_grad()
                        outputs,KD_avg_ori_sisnr ,KD_avg_sisnr = utils.sisnr_KD_compute_loss(model,teacher_model, x,
                                                                                               targets, My_criterion,KD_rate,len(targets),compute_grad=True)
                        optimizer.step()

                        optimizer_model_for_backward.zero_grad()
                        _, _,_  = utils.sisnr_KD_compute_loss(model_for_backward,teacher_model, x, 
                                                                targets, My_criterion,0,len_batch,compute_grad=True)
                        optimizer_model_for_backward.step()

                        # calculate backwarded model MSE
                        after_KD_loss = utils.sisnr_loss_for_sample(model, x, targets,len_batch)
                        after_loss = utils.sisnr_loss_for_sample(model_for_backward, x, targets,len_batch)
                        

                        # calculate r
                        RL_reward=(after_KD_loss - after_loss)
                        avg_origin_loss+=sisnr_loss/batch_num

                        # backward RL 
                        optimizer_rl.zero_grad()
                        RL_loss=utils.RL_compute_loss(RL_alpha,RL_reward,nn.MSELoss())
                        optimizer_rl.step()

                        # avg_r
                        avg_reward=torch.mean(RL_reward)
                        total_RL_reward+=avg_reward.item()
                        
                        # modify alpha_memory
                        if len_batch==args.batch_size:
                            alpha_memory[example_num]=KD_rate
                        else :
                            alpha_memory_final=KD_rate
                        # print info
                        print(f'avg_KD_rate = {avg_KD_rate} avg_ori_KD_rate={avg_ori_KD_rate}')
                        print(f'origin_loss        = {origin_loss}')
                        print(f'student_KD_loss    = {np.mean(after_KD_loss.cpu().detach().numpy())}')
                        print(f'student_copy_loss  = {np.mean(after_loss.cpu().detach().numpy())}')
                        print(f'avg_reward         = {avg_reward}')
                        print(f'total_RL_reward    = {total_RL_reward}')
                        print(f'RL_loss            = {RL_loss}')
                        # add to tensorboard
                        writer.add_scalar("train_student_avg_loss", student_avg_loss, state["step"])
                        writer.add_scalar("train_student_total_avg_loss", student_total_avg_loss, state["step"])
                        writer.add_scalar("origin_loss", origin_loss, state["step"])
                        writer.add_scalar("student_KD_loss", np.mean(after_KD_loss.cpu().detach().numpy()), state["step"])
                        writer.add_scalar("distillation_loss", dis_loss, state["step"])
                        writer.add_scalar("avg_KD_rate",avg_KD_rate, state["step"])
                        writer.add_scalar("RL_loss",RL_loss, state["step"])
                        writer.add_scalar("RL_reward", avg_reward, state["step"])

                    else: # no KD training
                        utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        
                        optimizer.zero_grad()
                        outputs, sisnr_loss = utils.sisnr_compute_loss(model, x, targets,len_batch, compute_grad=True)
                        optimizer.step()

                        avg_origin_loss+=sisnr_loss/batch_num
                        writer.add_scalar("origin_loss", sisnr_loss, state["step"])
                   

                    
                       

                    if example_num % args.example_freq == 0:
                        input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                        
                        target=torch.mean(targets[0], 0).cpu().numpy()
                        pred=torch.mean(outputs[0], 0).detach().cpu().numpy()
                        inputs=input_centre.cpu().numpy()


                        writer.add_audio("input:", input_centre, state["step"], sample_rate=args.sr)
                        writer.add_audio("pred:", torch.mean(outputs[0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio("target", torch.mean(targets[0], 0), state["step"], sample_rate=args.sr)

                    state["step"] += 1
                    pbar.update(1)



            all_avg_KD_rate=torch.mean(alpha_memory).item()
            print(f'all_avg_KD_rate:{all_avg_KD_rate}') # 之後再把final 加進來平均


            # VALIDATE
            val_loss,val_metrics = validate(args, model, criterion, val_data)
            print("ori VALIDATION FINISHED: LOSS: " + str(val_loss))
            
            
            writer.add_scalar("avg_origin_loss", avg_origin_loss, state["epochs"])
            choose_val=0
            
            if args.teacher_model is not None :
                val_loss_copy,val_metrics_copy = validate(args, model_for_backward, criterion, val_data)
                print("copy VALIDATION FINISHED: LOSS: " + str(val_loss_copy))
                
                if val_metrics[0]>val_metrics_copy[0]:
                    KD_to_copy=True
                    choose_val=val_metrics
                else:
                    KD_to_copy=False
                    choose_val=val_metrics_copy


                if KD_to_copy is True:
                    print('RL choose is better ')
                    # model_for_backward.load_state_dict(copy.deepcopy(model.state_dict()))
                    # optimizer_model_for_backward.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                else:
                    print('copy is better')
                    # model.load_state_dict(copy.deepcopy(model_for_backward.state_dict()))
                    # optimizer.load_state_dict(copy.deepcopy(optimizer_model_for_backward.state_dict()))

                for i in range(len(KD_rate)):
                    writer.add_scalar("KD_rate_"+str(i), KD_rate[i], state["epochs"])

                writer.add_scalar("all_avg_KD_rate", all_avg_KD_rate, state["epochs"])
                writer.add_scalar("val_loss_copy", val_loss_copy, state["epochs"])
                writer.add_scalar("total_RL_reward", total_RL_reward, state["epochs"])
                

                RL_checkpoint_path = os.path.join(args.checkpoint_dir, "RL_checkpoint_" + str(state["epochs"]))
                utils.save_model(rl, optimizer_rl, state, RL_checkpoint_path)
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
            utils.save_model(model, optimizer, state, checkpoint_path)
            state["epochs"] += 1
    if args.test is False:
        writer.close()
    #### TESTING ####
    # Test loss
    print("TESTING")
    # eval metrics
    test_metrics = evaluate(args, dataset["test"], model)
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

    



if __name__ == '__main__':
    ## TRAIN PARAMETERS
    model_base_path='/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/model'
    model_name = "down_size_student_myKD114"
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
