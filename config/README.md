
## cuda
* Use CUDA 
## outside_test
* testing model with outside_test
## myKD_method
* if set it to false,KD method is A plan (1-alpha,alpha) , else B plan (1,alpha) (default: False)
## KD_to_copy
* KD make a copy to copy
## num_workers
* Number of data loader worker threads 
## features
* Number of feature channels per layer
## test_dir
* Folder to write logs into
## dataset_dir
* Dataset path
## hdf_dir
* hdf_dir path
## teacher_model
* load a  pre-trained teacher model
## load_model
* Reload a previously trained student model 
## load_RL_model
* Reload a previously trained policy netwrok model 
## lr
* Initial learning rate in LR cycle 
## RL_lr
* 'Initial RL_learning rate in LR cycle 
## min_lr
* Minimum learning rate in LR cycle 
## cycles
* Number of LR cycles per epoch
## batch_size
* Batch size
## levels
* Number of DS/US blocks 
## levels_without_sample
* Number of non-updown-DS/US blocks 
## depth
* Number of convs per block
## sr
* Sampling rate
## decayRate
* decayRate for policy network
## channels
* Number of input audio channels
## encoder_kernel_size
* Filter width of kernels. Has to be an odd number
## decoder_kernel_size
* Filter width of kernels. Has to be an odd number
## output_size
* Output duration
## strides
* Strides in Waveunet
## example_freq
* Write an audio summary into Tensorboard logs every X training iterations
## loss
* L1 or L2
## conv_type
* Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn
## res
* Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned
## feature_growth
* Use CUDA (default: False)
## write_to_wav
* test with write wav
## output
* write wav dist path 
                            