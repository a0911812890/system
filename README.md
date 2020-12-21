
# system
# 新增 
1. 訓練時 保存參數 在 args.csv  

# 修改
1. 模型保存方式 -> 訓練時會直接開/model_name 保存相關資訊 
2. 現在下--teacher_model 即可做KD訓練 移除args.KD 參數

# note:
1. 請現在 train.py 設定 model_name 確保不會覆蓋 舊模型
2. 下--test 請把model_name改為test
