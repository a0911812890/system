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
from test import evaluate, validate,ling_evaluate,evaluate_without_noisy
from waveunet import Waveunet
from RL import RL
from Memory import Memory
from Loss import customLoss,RL_customLoss
#
    

def main(args):
    torch.cuda.manual_seed_all(1)
    np.random.seed(0)
    

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
        print(f'KD_method  = {args.KD_method}')
        
        with open(os.path.join(log_dir,'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=5)
        print('saving commandline_args')

        if args.teacher_model is not None:
            teacher_num_features = [24*i for i in range(1, args.levels+2+args.levels_without_sample)] 
            teacher_model = Waveunet(args.channels, teacher_num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)
            
            student_copy = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                        encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                        target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                        conv_type=args.conv_type, res=args.res)
            copy_optimizer = Adam(params=student_copy.parameters(), lr=args.lr)

            student_copy2 = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                        encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                        target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                        conv_type=args.conv_type, res=args.res)
            copy2_optimizer = Adam(params=student_copy2.parameters(), lr=args.lr)

            policy_network=RL(n_inputs=2,kernel_size=6,stride=1,conv_type=args.conv_type,pool_size=4)
            PG_optimizer = Adam(params=policy_network.parameters(), lr=args.RL_lr)
            if args.cuda:
                teacher_model = utils.DataParallel(teacher_model)
                policy_network = utils.DataParallel(policy_network)
                student_copy = utils.DataParallel(student_copy)
                student_copy2 = utils.DataParallel(student_copy2)
                # print("move teacher to gpu\n")
                teacher_model.cuda()
                # print("student_copy  to gpu\n")
                student_copy.cuda()
                # print("student_copy2  to gpu\n")
                student_copy2.cuda()
                # print("move policy_network to gpu\n")
                policy_network.cuda()
            student_size = sum(p.numel() for p in student_KD.parameters())
            teacher_size=sum(p.numel() for p in teacher_model.parameters())
            print('student_parameter count: ', str(student_size))
            print('teacher_model_parameter count: ', str(teacher_size))
            print('RL_parameter count: ', str(sum(p.numel() for p in policy_network.parameters())))
            print(f'compression raito :{100*(student_size/teacher_size)}%')
            if args.teacher_model is not None :
                print("load teacher model" + str(args.teacher_model))
                _ = utils.load_model(teacher_model, None, args.teacher_model, args.cuda)
                teacher_model.eval()
                
            if args.load_RL_model is not None:
                print("Continuing full RL_model from checkpoint " + str(args.load_RL_model))
                _ = utils.load_model(policy_network, PG_optimizer, args.load_RL_model, args.cuda)

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
        if state["epochs"]>0:
            state["epochs"]=state["epochs"]+1
        batch_num=(len(train_data) // args.batch_size)
        
        if args.teacher_model is not None :
            counting=0
            PG_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=PG_optimizer, gamma=args.decayRate)
            
            while counting<state["epochs"]:
                PG_optimizer.zero_grad()
                PG_optimizer.step()
                counting+=1
                PG_lr_scheduler.step()
            # print(f'modify lr RL rate : {counting} , until : {state["epochs"]}')
        while state["epochs"] < 100:
            memory_alpha=[]
            print("epoch:"+str(state["epochs"]))
            student_KD.train()
            student_copy.train()
            student_copy2.train()
            # monitor_value    
            total_avg_reward=0
            total_avg_scalar_reward=0
            avg_origin_loss=0
            all_avg_KD_rate=0
            same=0
            with tqdm(total=len(dataloader)) as pbar:
                for example_num, (x, targets) in enumerate(dataloader):
                    # if example_num==20:
                    #     break
                    if args.cuda:
                        x = x.cuda()
                        targets = targets.cuda()
                    if args.teacher_model is not None:
                        # Set LR for this iteration  
                        temp =  {
                            'state_dict' : None,
                            'optim_dict' : None
                        }

                        
                        temp['state_dict']=copy.deepcopy(student_KD.state_dict())
                        temp['optim_dict']=copy.deepcopy(KD_optimizer.state_dict())
                        #print('base_model from KD')

                        student_KD.load_state_dict(temp['state_dict'])
                        KD_optimizer.load_state_dict(temp['optim_dict'])

                        student_copy.load_state_dict(temp['state_dict'])
                        copy_optimizer.load_state_dict(temp['optim_dict'])

                        student_copy2.load_state_dict(temp['state_dict'])
                        copy2_optimizer.load_state_dict(temp['optim_dict'])

                        utils.set_cyclic_lr(KD_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        utils.set_cyclic_lr(copy_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        utils.set_cyclic_lr(copy2_optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                        # forward student and teacher  get output
                        student_KD_output, avg_student_KD_loss=utils.compute_loss(student_KD, x, targets, criterion,compute_grad=False)
                        teacher_output, _=utils.compute_loss(teacher_model, x, targets, criterion,compute_grad=False)
                        # PG_state
                        diff_from_target=targets.detach()-student_KD_output.detach()
                        diff_from_teacher=teacher_output.detach()-student_KD_output.detach()
                        PG_state=torch.cat((diff_from_target,diff_from_teacher),1)

                        # forward RL get alpha
                        alpha=policy_network(PG_state)
                        nograd_alpha=alpha.detach()
                        
                        avg_KD_rate=torch.mean(nograd_alpha).item()
                        all_avg_KD_rate+=avg_KD_rate / batch_num

                        KD_optimizer.zero_grad()
                        KD_outputs, KD_hard_loss ,KD_loss ,KD_soft_loss = utils.KD_compute_loss(student_KD,teacher_model, x, targets, My_criterion,alpha=nograd_alpha,compute_grad=True,KD_method=args.KD_method)
                        KD_optimizer.step()

                        copy_optimizer.zero_grad()
                        _,_,_,_ = utils.KD_compute_loss(student_copy,teacher_model, x, targets, My_criterion,alpha=1,compute_grad=True,KD_method=args.KD_method)
                        copy_optimizer.step()

                        copy2_optimizer.zero_grad()
                        _,_,_,_ = utils.KD_compute_loss(student_copy2,teacher_model, x, targets, My_criterion,alpha=0,compute_grad=True,KD_method=args.KD_method)
                        copy2_optimizer.step()

                        # calculate backwarded model MSE
                        backward_KD_loss = utils.loss_for_sample(student_KD, x, targets)
                        backward_copy_loss = utils.loss_for_sample(student_copy, x, targets)
                        backward_copy2_loss = utils.loss_for_sample(student_copy2, x, targets)

                        # calculate rewards
                        rewards,same_num,before_decay=utils.get_rewards(backward_KD_loss,backward_copy_loss,backward_copy2_loss,backward_KD_loss,len(train_data),state["epochs"]+1)
                        same+= same_num

                        avg_origin_loss += avg_student_KD_loss / batch_num
                        

                        # avg_reward
                        avg_reward=torch.mean(rewards)
                        avg_scalar_reward=torch.mean(torch.abs(rewards))
                        total_avg_reward+=avg_reward.item()/batch_num
                        total_avg_scalar_reward+=avg_scalar_reward.item()/batch_num
                        # append to memory_alpha
                        nograd_alpha = nograd_alpha.detach().cpu()
                        memory_alpha.append(nograd_alpha.numpy())

                        PG_optimizer.zero_grad()
                        _=utils.RL_compute_loss(alpha,rewards,nn.MSELoss())
                        PG_optimizer.step()
                        # print info
                        # print(f'avg_KD_rate                 = {avg_KD_rate} ')
                        # print(f'student_KD_loss             = {avg_student_KD_loss}')              
                        # print(f'backward_student_copy_loss  = {np.mean(backward_copy_loss.detach().cpu().numpy())}')
                        # print(f'backward_student_KD_loss    = {np.mean(backward_KD_loss.detach().cpu().numpy())}')
                        # print(f'backward_student_copy2_loss = {np.mean(backward_copy2_loss.detach().cpu().numpy())}')
                        # print(f'avg_reward                  = {avg_reward}')
                        # print(f'total_avg_reward            = {total_avg_reward}')
                        # print(f'same                        = {same}')

                        # add to tensorboard
                        writer.add_scalar("student_KD_loss", avg_student_KD_loss, state["step"])
                        writer.add_scalar("backward_student_KD_loss", np.mean(backward_KD_loss.detach().cpu().numpy()), state["step"])
                        writer.add_scalar("KD_loss", KD_loss, state["step"])
                        writer.add_scalar("KD_hard_loss", KD_hard_loss, state["step"])
                        writer.add_scalar("KD_soft_loss", KD_soft_loss, state["step"])
                        writer.add_scalar("avg_KD_rate",avg_KD_rate, state["step"])
                        writer.add_scalar("rewards", avg_reward, state["step"])
                        writer.add_scalar("scalar_rewards", avg_scalar_reward, state["step"])
                        writer.add_scalar("before_decay", before_decay, state["step"])
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
                        
                        # target=torch.mean(targets[0], 0).cpu().numpy()
                        # pred=torch.mean(KD_outputs[0], 0).detach().cpu().numpy()
                        # inputs=input_centre.cpu().numpy()

                        writer.add_audio("input:", input_centre, state["step"], sample_rate=args.sr)
                        writer.add_audio("pred:", torch.mean(KD_outputs[0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio("target", torch.mean(targets[0], 0), state["step"], sample_rate=args.sr)

                    state["step"] += 1
                    pbar.update(1)
            # VALIDATE
            val_loss,val_metrics = validate(args, student_KD, criterion, val_data)
            print("ori VALIDATION FINISHED: LOSS: " + str(val_loss))


            choose_val=val_metrics
            if args.teacher_model is not None :
                for i in range(len(nograd_alpha)):
                    writer.add_scalar("KD_rate_"+str(i), nograd_alpha[i], state["epochs"])
                print(f'all_avg_KD_rate = {all_avg_KD_rate}')
                writer.add_scalar("all_avg_KD_rate", all_avg_KD_rate, state["epochs"])
                # writer.add_scalar("val_loss_copy", val_loss_copy, state["epochs"])
                writer.add_scalar("total_avg_reward", total_avg_reward, state["epochs"])
                writer.add_scalar("total_avg_scalar_reward", total_avg_scalar_reward, state["epochs"])
                
                RL_checkpoint_path = os.path.join(checkpoint_dir, "RL_checkpoint_" + str(state["epochs"]))
                utils.save_model(policy_network, PG_optimizer, state, RL_checkpoint_path)
                PG_lr_scheduler.step()
                
                
            writer.add_scalar("same", same, state["epochs"])
            writer.add_scalar("avg_origin_loss", avg_origin_loss, state["epochs"])
            writer.add_scalar("val_enhance_pesq",choose_val[0], state["epochs"])
            writer.add_scalar("val_improve_pesq",choose_val[1], state["epochs"])
            writer.add_scalar("val_enhance_stoi",choose_val[2], state["epochs"])
            writer.add_scalar("val_improve_stoi",choose_val[3], state["epochs"])
            writer.add_scalar("val_enhance_SISDR",choose_val[4], state["epochs"])
            writer.add_scalar("val_improve_SISDR",choose_val[5], state["epochs"])
           # writer.add_scalar("val_COPY_pesq",val_metrics_copy[0], state["epochs"])
            writer.add_scalar("val_loss", val_loss, state["epochs"])

            # Set up training state dict that will also be saved into checkpoints
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_" + str(state["epochs"]))
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


    
        
    # test_data = SeparationDataset(dataset, "test", args.sr, args.channels, student_KD.shapes, False, args.hdf_dir, audio_transform=crop_func)
    
        
    

    #### TESTING ####
    # Test loss
    print("TESTING")
    # eval metrics
    #ling_data=get_ling_data_list('/media/hd03/sutsaiwei_data/data/mydata/ling_data')
    #validate(args, student_KD, criterion, test_data)
    #test_metrics = ling_evaluate(args, ling_data['noisy'], student_KD)
    #test_metrics = evaluate_without_noisy(args, dataset["test"], student_KD)
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
        #print(args.__dict__)
    ## loading configs.json
    


    main(args)
