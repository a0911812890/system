
# system
1. gen_data.py  
    依靠speech , noise 產生 noisy
2. data.py
    根據模型輸入，把 speech , noisy 切割成直接輸入可用 並存成 test.hdf 和 train.hdf
    資料處理函數
3. waveunet.py
    整體enhancement model架構
4. waveunet_utils.py
    enhancement model 各區塊
5. Loss.py
    重新定義loss使得可以單個sample乘上對應之alpha
6. oneMask_criterion.py 
    用來計算sisdr loss 
7. resmpleWav.py
    resample data
8. RL.py
    RL之架構
9. test.py
    關於模型測試 包刮 validate 及 evaluate
10. train.py 
    根據config 建構模型並訓練 (reinforcemnet learning-based KD 及 non KD )
11. oriKD_train.py
    根據config 建構模型並訓練 (fixed KD)

# config
1. config.json 
    模型根據config.json訓練

