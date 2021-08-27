
## system
train(test=false) 
終端機輸入:python train.py

會產生三個資料夾
1.checkpoints   :保存checkpoints
2.logs          :包含 alpha每次EPOCHS的值 config.json tensorboard值   
3.results       :實驗跑完時 選最好checkpoints(根據PESQ) 跑 test

test(test=true) 時 只需輸入model checkpoints(load_model) 測試完CSV檔放在~/model/test/results
終端機輸入:python train.py

如果test=true  ，會直接跑測試
如果test=false ，會先訓練再測試

## gen_data.py  
* 依靠speech , noise 產生 noisy 可調整 SNR
## data.py
* 根據模型輸入，把 speech , noisy 切割成直接輸入可用 並存成 test.hdf 和 train.hdf 資料處理函數
## waveunet.py
* 整體enhancement model架構 包含7個encoder(有down sample) + 5個encoder(無down sample) +1個bottleneck+7個decoder(up sample) + 5個decoder(無up sample) 
## waveunet_utils.py
* enhancement model 各區塊 基礎元件 
## Loss.py
* 重新定義loss使得可以單個sample乘上對應之alpha
## oneMask_criterion.py 
* 用來計算sisdr loss 
## resmpleWav.py
* resample data
## RL.py
* RL之架構
## test.py
* 關於模型測試 包刮 validate (訓練中測試) 及 evaluate(訓練完測試)
## train.py 
* 根據config 建構模型並訓練 (reinforcemnet learning-based KD 及 non KD )
## oriKD_train.py
* 根據config 建構模型並訓練 (fixed KD)

# config
1. config.json 
    模型根據config.json訓練
## cuda
* Use CUDA 
## outside_test
* testing model with outside_test 是否使用outside_test dataset (default:test)
## myKD_method
* if set it to false,KD method is A method (1-alpha,alpha) , else B method (1,alpha)  , C method (alpha,1)  [hard-label,soft-label]
## num_workers
* Number of data loader worker threads 
## features
* Number of feature channels per layer ###控制kernel size數量 (模型大小) 寫死 100% 約3天多 
## dataset_dir
* Data 的路徑 (寫死直接使用train,test) train及test都包含speech,noisy,noise
## hdf_dir
* hdf_dir path ### 預處理 把data 資料切成 模型輸入大小 並存起來 (如果沒有才會產生)
## teacher_model
* load a  pre-trained teacher model ### 絕對路徑讀取 checkpoint 
## load_model
* Reload a previously trained student model  ### 絕對路徑讀取 checkpoint 
## load_RL_model
* Reload a previously trained policy netwrok model ### 絕對路徑讀取 checkpoint 
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
* Number of DS/US blocks  ### 7層 設為 stride 為 2 (up-down sample)
## levels_without_sample
* Number of non-updown-DS/US blocks 5層 設為 stride 為 1
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
* Strides in Waveunet ### 7 (levels) up-down sample 縮減步長
## example_freq
* Write an audio summary into Tensorboard logs every X training iterations
## loss
* L1 or L2
## conv_type
* Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn
## res
* Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned
## write_to_wav
* test with write wav ### test完的結果是否輸出成wav
## output
* write wav dist path 
                            
