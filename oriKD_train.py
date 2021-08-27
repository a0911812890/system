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
import pickle
import json
#
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam,SGD
#
import utils 
from data import  SeparationDataset,  crop ,get_folds,get_ling_data_list
from test import evaluate, validate,ling_evaluate,evaluate_for_enhanced
from waveunet import Waveunet
from RL import RL
from Loss import customLoss,RL_customLoss
#
    

def main(args):
    os.environ['KMP_WARNINGS'] = '0'
    torch.cuda.manual_seed_all(1)
    np.random.seed(0)
    print(args.model_name)
    print(args.alpha)
    # filter array
    num_features = [args.features*i for i in range(1, args.levels+2+args.levels_without_sample)] 
    
    # 確定 輸出大小
    target_outputs = int(args.output_size * args.sr)
    # 訓練才保存模型設定參數


    # 設定teacher and student and student_for_backward 超參數

    student_KD = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
    KD_optimizer = Adam(params=student_KD.parameters(), lr=args.lr)
    print(25*'='+'model setting'+25*'=')
    print('student_KD: ', student_KD.shapes)
    if args.cuda:
        student_KD = utils.DataParallel(student_KD)
        print("move student_KD to gpu\n")
        student_KD.cuda()
        
    
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_pesq" : -np.Inf}
    if args.load_model is not None:
        print("Continuing full model from checkpoint " + str(args.load_model))
        state = utils.load_model(student_KD, KD_optimizer, args.load_model, args.cuda)
    dataset = get_folds(args.dataset_dir,args.outside_test)
    log_dir,checkpoint_dir,result_dir=utils.mkdir_and_get_path(args)
    # print(model)
    if args.test is False:
        writer = SummaryWriter(log_dir)
        # set hypeparameter
        # printing hypeparameters info
        print(25*'='+'printing hypeparameters info'+25*'=')
        
        
        with open(os.path.join(log_dir,'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=5)
        print('saving commandline_args')
        student_size = sum(p.numel() for p in student_KD.parameters())
        print('student_parameter count: ', str(student_size))
        if args.teacher_model is not None:
            teacher_num_features = [24*i for i in range(1, args.levels+2+args.levels_without_sample)] 
            teacher_model = Waveunet(args.channels, teacher_num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
            
            if args.cuda:
                teacher_model = utils.DataParallel(teacher_model)
                teacher_model.cuda()
                # print("move teacher to gpu\n")
            student_size = sum(p.numel() for p in student_KD.parameters())
            teacher_size=sum(p.numel() for p in teacher_model.parameters())
            print('student_parameter count: ', str(student_size))
            print('teacher_model_parameter count: ', str(teacher_size))
            print(f'compression raito :{100*(student_size/teacher_size)}%')
            if args.teacher_model is not None :
                print("load teacher model" + str(args.teacher_model))
                _ = utils.load_model(teacher_model, None, args.teacher_model, args.cuda)
                teacher_model.eval()
                

         # If not data augmentation, at least crop targets to fit model output shape
        crop_func = partial(crop, shapes=student_KD.shapes)
        ### DATASET
        train_data = SeparationDataset(dataset, "train", args.sr, args.channels, student_KD.shapes, False, args.hdf_dir, audio_transform=crop_func)
        val_data = SeparationDataset(dataset, "test", args.sr, args.channels, student_KD.shapes, False, args.hdf_dir, audio_transform=crop_func)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn,pin_memory=True)
        
        # Set up the loss function
        if args.loss == "L1":
            criterion = nn.L1Loss()
        elif args.loss == "L2":
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError("Couldn't find this loss!")
        My_criterion = customLoss()

        ### TRAINING START
        print('TRAINING START')
        batch_num=(len(train_data) // args.batch_size)
        while state["epochs"] < 100:
        #     if state["epochs"]<10:
        #         args.alpha=1
        #     else:
        #         args.alpha=0
            # print('fix alpha:',args.alpha)
            memory_alpha=[]
            print("epoch:"+str(state["epochs"]))
            student_KD.train()
            # monitor_value    
            avg_origin_loss=0
            with tqdm(total=len(dataloader)) as pbar:
                for example_num, (x, targets) in enumerate(dataloader):
                    if args.cuda:
                        x = x.cuda()
                        targets = targets.cuda()
                    if args.teacher_model is not None:
                        # Set LR for this iteration  
                        #print('base_model from KD')
                        

                        utils.set_cyclic_lr(KD_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        _, avg_student_KD_loss=utils.compute_loss(student_KD, x, targets, criterion,compute_grad=False)
                        
                        KD_optimizer.zero_grad()
                        KD_outputs, KD_hard_loss ,KD_loss ,KD_soft_loss = utils.KD_compute_loss(student_KD,teacher_model, x, targets, My_criterion,alpha=args.alpha,compute_grad=True,KD_method=args.KD_method)
                        KD_optimizer.step()


                        # calculate backwarded model MSE
                        
                        avg_origin_loss += avg_student_KD_loss / batch_num
                        
                        # add to tensorboard
                        writer.add_scalar("KD_loss", KD_loss, state["step"])
                        writer.add_scalar("KD_hard_loss", KD_hard_loss, state["step"])
                        writer.add_scalar("KD_soft_loss", KD_soft_loss, state["step"])
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

                        writer.add_audio("input:", input_centre, state["step"], sample_rate=args.sr)
                        writer.add_audio("pred:", torch.mean(KD_outputs[0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio("target", torch.mean(targets[0], 0), state["step"], sample_rate=args.sr)

                    state["step"] += 1
                    pbar.update(1)
            # VALIDATE
            val_loss,val_metrics = validate(args, student_KD, criterion, val_data)
            print("ori VALIDATION FINISHED: LOSS: " + str(val_loss))


            writer.add_scalar("avg_origin_loss", avg_origin_loss, state["epochs"])
            writer.add_scalar("val_enhance_pesq",val_metrics[0], state["epochs"])
            writer.add_scalar("val_improve_pesq",val_metrics[1], state["epochs"])
            writer.add_scalar("val_enhance_stoi",val_metrics[2], state["epochs"])
            writer.add_scalar("val_improve_stoi",val_metrics[3], state["epochs"])
            writer.add_scalar("val_enhance_SISDR",val_metrics[4], state["epochs"])
            writer.add_scalar("val_improve_SISDR",val_metrics[5], state["epochs"])
           # writer.add_scalar("val_COPY_pesq",val_metrics_copy[0], state["epochs"])
            writer.add_scalar("val_loss", val_loss, state["epochs"])

            # Set up training state dict that will also be saved into checkpoints
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_" + str(state["epochs"]))
            if val_metrics[0] < state["best_pesq"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_pesq"] = val_metrics[0]
                state["best_checkpoint"] = checkpoint_path

            # CHECKPOINT
            print("Saving model...")
            utils.save_model(student_KD, KD_optimizer, state, checkpoint_path)
            print('dump alpha_memory')
            with open(os.path.join(log_dir, 'alpha_'+str(state["epochs"])), "wb") as fp:   #Pickling
                pickle.dump(memory_alpha, fp)
            state["epochs"] += 1
        writer.close()
        info=args.model_name
        path=os.path.join(result_dir,info)
    else:
        PATH=args.load_model.split("/")
        info=PATH[-3]+"_"+PATH[-1]
        if(args.outside_test==True):
            info+="_outside_test"
        print(info)
        path=os.path.join(result_dir,info)
    
    #### TESTING ####
    # Test loss
    print("TESTING")
    # eval metrics
    _ = utils.load_model(student_KD, KD_optimizer, state["best_checkpoint"], args.cuda)
    test_metrics = evaluate(args, dataset["test"], student_KD)
    test_pesq=test_metrics['pesq']
    test_stoi=test_metrics['stoi']
    test_SISDR=test_metrics['SISDR']
    test_noise=test_metrics['noise']

    
    if not os.path.exists(path):
        os.makedirs(path)
    utils.save_result(test_pesq,path,"pesq")
    utils.save_result(test_stoi,path,"stoi")
    utils.save_result(test_SISDR,path,"SISDR")
    utils.save_result(test_noise,path,"noise")

if __name__ == '__main__':
    ## loading configs.json
    basepath=os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    with open(os.path.join(basepath,'config','config.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    


    main(args)
