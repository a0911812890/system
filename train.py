import argparse
import os
from functools import partial
import pandas as pd
import numpy as np
import time
from datetime import date,datetime
from tqdm import tqdm
from pypesq import pesq
#
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
#
import utils
from data import  SeparationDataset,  crop ,get_folds
from test import evaluate, validate
from waveunet import Waveunet
from RL import RL
#

def main(args):
    #torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5
    # filter array
    num_features = [args.features*i for i in range(1, args.levels+2+args.levels_without_sample)] 
    
    # 確定 輸出大小
    target_outputs = int(args.output_size * args.sr)
    # 訓練才保存模型設定參數
    if args.test is False:
        utils.args_to_csv(args)


    # 設定teacher and student and student_for_backward 超參數
    #teacher_model=0
    #model_for_backward=0

    model = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
    if args.cuda:
        model = utils.DataParallel(model)
        print("move model to gpu\n")
        model.cuda()
    

    if args.teacher_model is not None:
        teacher_model = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                conv_type=args.conv_type, res=args.res)
        if args.cuda:
            teacher_model = utils.DataParallel(teacher_model)
            print("move teacher_model to gpu\n")
            teacher_model.cuda()
        
        model_for_backward = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
        if args.cuda:
            model_for_backward = utils.DataParallel(model_for_backward)
            print("model_for_backward model to gpu\n")
            model_for_backward.cuda()

        rl=RL(n_inputs=3,kernel_size=6,stride=1,conv_type=args.conv_type,pool_size=4)
        if args.cuda:
            rl = utils.DataParallel(rl)
            print("move rl to gpu\n")
            rl.cuda()
   
    # print('model: ', model.shapes)
    # print('student_parameter count: ', str(sum(p.numel() for p in model.parameters())))
    # print('RL_parameter count: ', str(sum(p.numel() for p in rl.parameters())))
    # print(model)
    writer = SummaryWriter(args.log_dir)

    # ### DATASET

    dataset = get_folds(args.dataset_dir)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop, shapes=model.shapes)
    # Data augmentation function for training
    if args.test is False:
        train_data = SeparationDataset(dataset, "train", args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=crop_func)
        val_data = SeparationDataset(dataset, "val", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    test_data = SeparationDataset(dataset, "test", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)

        

    # ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    if args.teacher_model is not None:
        optimizer_model_for_backward = Adam(params=model_for_backward.parameters(), lr=args.lr)
        #optimizer_rl = Adam(params=rl.parameters(), lr=args.lr)
        optimizer_rl = Adam(params=rl.parameters(), lr=0.0001)
    
    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_RL_model is not None:
        print("Continuing full model from checkpoint " + str(args.load_RL_model))
        rl.load_state_dict(torch.load(args.load_RL_model))
    if args.load_model is not None:
        print("Continuing full model from checkpoint " + str(args.load_model))
        state = utils.load_model(model, optimizer, args.load_model, args.cuda)

    # load teacher model
    if args.teacher_model is not None :
        print("load teacher model" + str(args.teacher_model))
        teacher_state = utils.load_model(teacher_model, None, args.teacher_model, args.cuda)
        teacher_model.eval()

    if args.test is False:
        print('TRAINING START')
        KD_to_copy=True
        while state["epochs"] < 150:#state["worse_epochs"] < args.patience  and 
            print("epoch:"+str(state["epochs"]))
            print("Training one epoch from iteration " + str(state["step"]))
            avg_time = 0.
            model.train()
            total_RL_reward=0
            with tqdm(total=len(train_data) // args.batch_size) as pbar:
                np.random.seed()
                for example_num, (x, targets) in enumerate(dataloader):
                    if args.cuda:
                        x = x.cuda()
                        targets = targets.cuda()

                    t = time.time()

                    # Set LR for this iteration
                    utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                    utils.set_cyclic_lr(optimizer_model_for_backward, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                    #utils.set_cyclic_lr(optimizer_rl, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                    writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])
                    # Compute loss for model    
                    if args.teacher_model is not None:
                        # copy s_model to backward_model (At the beginning in a batch, s_model equal to b_model ) 
                        if KD_to_copy is True:
                            model_for_backward.load_state_dict(model.state_dict())
                            
                        else:
                            model.load_state_dict(model_for_backward.state_dict())
                        # forward student and teacher  get output
                        student_output, _=utils.compute_loss(model, x, targets, criterion,compute_grad=False)
                        teacher_output, _=utils.compute_loss(teacher_model, x, targets, criterion,compute_grad=False)
                        # concat s_out,t_out ,target
                        rl_input=torch.cat((student_output,teacher_output,targets),1)
                        # forward RL get RL_alpha
                        RL_alpha=rl(rl_input)
                        # print(rl_output)
                        RL_alpha=torch.mean(RL_alpha,0).cpu().detach().numpy()
                        RL_alpha=round(RL_alpha[0],4)
                        # backward with alpha and without alpha 
                        optimizer.zero_grad()
                        outputs, student_avg_loss ,student_total_avg_loss  = utils.KD_compute_loss(model,teacher_model, x, targets, criterion,alpha=RL_alpha,compute_grad=True)
                        optimizer.step()

                        optimizer_model_for_backward.zero_grad()
                        _, normal_loss ,_  = utils.KD_compute_loss(model_for_backward,teacher_model, x, targets, criterion,alpha=0,compute_grad=True)
                        optimizer_model_for_backward.step()
                        
                        
                        # compute reward
                        _, after_KD_loss=utils.compute_loss(model, x, targets, criterion,compute_grad=False)
                        _, after_loss=utils.compute_loss(model_for_backward, x, targets, criterion,compute_grad=False)
                        RL_reward=after_loss - after_KD_loss
                        

                        # backward RL 
                        optimizer_rl.zero_grad()
                        # RL_loss=utils.RL_compute_loss(rl_output,RL_reward,nn.CrossEntropyLoss(),args.batch_size)
                        RL_loss,sign_normalize_reward=utils.RL_compute_loss(RL_alpha_array,RL_reward,criterion)
                        optimizer_rl.step()
    
                        total_RL_reward+=sign_normalize_reward
                        # if RL_alpha>0.01:
                        #     KD_to_copy=True
                        #     print('KD_to_copy')
                        # else:
                        #     KD_to_copy=False
                        #     print('copy_to_KD')
                        print(f'RL_alpha={RL_alpha}')
                        print(f'RL_reward={sign_normalize_reward}')
                        print(f'total_RL_reward={total_RL_reward}')
                        print(f'RL_loss={RL_loss}')
                    else:
                        optimizer.zero_grad()
                        outputs, student_avg_loss = utils.compute_loss(model, x, targets, criterion, compute_grad=True)
                        optimizer.step()

                   

                    state["step"] += 1

                    t = time.time() - t
                    avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                    if args.teacher_model is not None:
                        writer.add_scalar("train_student_avg_loss", student_avg_loss, state["step"])
                        writer.add_scalar("train_student_total_avg_loss", student_total_avg_loss, state["step"])
                        writer.add_scalar("RL_alpha",RL_alpha, state["step"])
                        writer.add_scalar("RL_reward", RL_reward, state["step"])
                        
                    else:
                        writer.add_scalar("train_student_avg_loss", student_avg_loss, state["step"])

                    if example_num % args.example_freq == 0:
                        input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                        
                        target=torch.mean(targets[0], 0).cpu().numpy()
                        pred=torch.mean(outputs[0], 0).detach().cpu().numpy()
                        inputs=input_centre.cpu().numpy()

                        values1=round(pesq(target, inputs,16000),2)
                        values2=round(pesq(target,pred ,16000),2)

                        writer.add_scalar("pesq_improve", values2 - values1, state["step"])

                        writer.add_audio("input:", input_centre, state["step"], sample_rate=args.sr)
                        writer.add_audio("pred:", torch.mean(outputs[0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio("target", torch.mean(targets[0], 0), state["step"], sample_rate=args.sr)
                    
                    pbar.update(1)
            print(f'total_RL_reward={total_RL_reward}')
            # VALIDATE
            val_loss = validate(args, model, criterion, val_data,writer,state)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))
            writer.add_scalar("val_loss", val_loss, state["epochs"])
            writer.add_scalar("total_RL_reward", total_RL_reward, state["epochs"])
            # EARLY STOPPING CHECK
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["epochs"]))
            RL_checkpoint_path = os.path.join(args.checkpoint_dir, "RL_checkpoint_" + str(state["epochs"]))
            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

            # CHECKPOINT
            print("Saving model...")
            utils.save_model(model, optimizer, state, checkpoint_path)
            state["epochs"] += 1
            torch.save(rl.state_dict(), RL_checkpoint_path)
    #### TESTING ####
    # Test loss
    print("TESTING")
    # Load best model based on validation loss
    # if args.test is None:
    #     state = utils.load_model(model, None, state["best_checkpoint"], args.cuda)

    test_loss = validate(args, model, criterion, test_data,writer,state)
    print("TEST FINISHED: LOSS: " + str(test_loss))
    writer.add_scalar("test_loss", test_loss, state["step"])

    # Mir_eval metrics
    test_metrics = evaluate(args, dataset["test"], model)
    test_pesq=test_metrics['pesq']
    test_stoi=test_metrics['stoi']
    # # Dump all metrics results into pickle and csv file for later analysis if needed

    date=datetime.now()
    date_str=date.strftime("%Y_%m_%d %H_%M_%S")
    print(date_str)
    time_dir=str(date_str)
    path=os.path.join(args.result_dir,time_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    utils.save_result(test_pesq,path,"pesq")
    utils.save_result(test_stoi,path,"stoi")
    
    writer.close()



if __name__ == '__main__':
    ## TRAIN PARAMETERS
    model_base_path='/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/model'
    model_name = "myKD_RL_normalize(lr10e4_init)"
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--test', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader worker threads (default: 4)')
    parser.add_argument('--features', type=int, default=24,
                        help='Number of feature channels per layer')
    parser.add_argument('--test_dir', type=str, default="/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/results",
                        help='Folder to write logs into')
    parser.add_argument('--log_dir', type=str, default=os.path.join(model_base_path,model_name,'logs'),
                        help='Folder to write logs into')
    parser.add_argument('--result_dir', type=str, default=os.path.join(model_base_path,model_name,'results'),
                        help='Folder to write results into')
    parser.add_argument('--dataset_dir', type=str, default="/media/hd03/sutsaiwei_data/data/yunwen_data",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/hdf/snr_hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(model_base_path,model_name,'checkpoints'),
                        help='Folder to write checkpoints into')
    parser.add_argument('--teacher_model', type=str, default=None,#"/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/backup/2020_snr_unet_origin/checkpoints/checkpoint_33034"
                        help='load a  pre-trained teacher model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--load_RL_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
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
    
    
    if not os.path.isdir(args.log_dir):
        os.makedirs( args.log_dir )
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs( args.checkpoint_dir )
    if not os.path.isdir(args.result_dir):
        os.makedirs( args.result_dir )
    main(args)
