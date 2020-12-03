#!/bin/bash

for audio in `ls /media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/audio_examples/test2` ;do
    python predict.py --load_model checkpoints/waveunet/checkpoint_235144 --input "/media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch/audio_examples/test2/$audio" --output /media/hd03/sutsaiwei_data/Wave-U-Net-Pytorch
done