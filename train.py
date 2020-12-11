import argparse
import os
import time
from functools import partial
import pandas as pd
import torch
import pickle
import numpy as np

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import utils
from data import get_musdb_folds, SeparationDataset, random_amplify, crop ,get_folds
from test import evaluate, validate
from waveunet import Waveunet
from pypesq import pesq
def main(args):
    #torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5
    # MODEL
    KD = True
    num_features = [args.features*i for i in range(1, args.levels+2+args.levels_without_sample)] 
    # print(num_features)
    target_outputs = int(args.output_size * args.sr)
    print(args.test)
    if args.test is False:
        print('OK')
    model = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)

    teacher_model = Waveunet(args.channels, num_features, args.channels,levels=args.levels, 
                    encoder_kernel_size=args.encoder_kernel_size,decoder_kernel_size=args.decoder_kernel_size,
                    target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                    conv_type=args.conv_type, res=args.res)

    if args.cuda:
        model = utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()
        # 
        teacher_model = utils.DataParallel(teacher_model)
        print("move teacher_model to gpu")
        teacher_model.cuda()

    # print('model: ', model.shapes)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))
    # print(model)

    # exit(0)
    writer = SummaryWriter(args.log_dir)

    # ### DATASET

    dataset = get_folds(args.dataset_dir)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop, shapes=model.shapes)
    # Data augmentation function for training
    if args.test is False:
        train_data = SeparationDataset(dataset, "train", args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=crop_func)
        val_data = SeparationDataset(dataset, "val", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
    test_data = SeparationDataset(dataset, "test", args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)

    if args.test is False:
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

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

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # load teacher model
    if KD :
        print("load teacher model" + str(args.teacher_model))
        teacher_state = utils.load_model(teacher_model, None, args.teacher_model, args.cuda)
        teacher_model.eval() # no training

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing full model from checkpoint " + str(args.load_model))
        state = utils.load_model(model, optimizer, args.load_model, args.cuda)

    if args.test is False:
        print('TRAINING START')
        while state["worse_epochs"] < args.patience:
            print("epoch:"+str(state["epochs"]))
            print("Training one epoch from iteration " + str(state["step"]))
            avg_time = 0.
            model.train()
            with tqdm(total=len(train_data) // args.batch_size) as pbar:
                np.random.seed()
                for example_num, (x, targets) in enumerate(dataloader):
                    if args.cuda:
                        x = x.cuda()
                        targets = targets.cuda()

                    t = time.time()

                    # Set LR for this iteration
                    utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                    writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                    # Compute loss for model
                    optimizer.zero_grad()
                    outputs, avg_loss ,KD_avg_loss ,total_avg_loss = utils.KD_compute_loss(model,teacher_model, x, targets, criterion, compute_grad=True)

                    optimizer.step()

                    state["step"] += 1

                    t = time.time() - t
                    avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                    writer.add_scalar("train_avg_loss", avg_loss, state["step"])
                    writer.add_scalar("train_KD_avg_loss", KD_avg_loss, state["step"])
                    writer.add_scalar("train_total_avg_loss", total_avg_loss, state["step"])

                    if example_num % args.example_freq == 0:
                        input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                        
                        target=torch.mean(targets[0], 0).cpu().numpy()
                        pred=torch.mean(outputs[0], 0).detach().cpu().numpy()
                        inputs=input_centre.cpu().numpy()

                        values1=round(pesq(target, inputs,16000),2)
                        writer.add_scalar("pesq_input", values1, state["step"])
                        values2=round(pesq(target,pred ,16000),2)
                        writer.add_scalar("pesq_enhance", values2, state["step"])

                        writer.add_scalar("pesq_improve", values2 - values1, state["step"])

                        writer.add_audio("input:", input_centre, state["step"], sample_rate=args.sr)
                        writer.add_audio("pred:", torch.mean(outputs[0], 0), state["step"], sample_rate=args.sr)
                        writer.add_audio("target", torch.mean(targets[0], 0), state["step"], sample_rate=args.sr)

                    pbar.update(1)

            # VALIDATE
            val_loss = validate(args, model, criterion, val_data,writer,state)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))
            writer.add_scalar("val_loss", val_loss, state["step"])

            # EARLY STOPPING CHECK
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
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

    # # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)
    data = pd.DataFrame(test_metrics)
    data.to_csv(os.path.join(args.checkpoint_dir, "results.csv"),sep=',')
    # writer.add_scalar("test_SDR", overall_SDR)

    writer.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--test', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader worker threads (default: 4)')
    parser.add_argument('--features', type=int, default=24,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='logs/snr_waveunet',
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/media/hd03/sutsaiwei_data/data/yunwen_data",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/hdf/snr_hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/checkpoints/snr_waveunet',
                        help='Folder to write checkpoints into')
    parser.add_argument('--teacher_model', type=str, default='/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/backup/2020_snr_unet_origin/checkpoints/checkpoint_33034',
                        help='load a  pre-trained teacher model')
    parser.add_argument('--load_model', type=str, default=None,
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

    main(args)
