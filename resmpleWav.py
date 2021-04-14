import os, librosa
import soundfile as sf

# 重新取樣
def resmple_wav(wav, target_sr):
    y, sr = librosa.load(wav)
    reampled_y = librosa.resample(y, sr, target_sr)
    return reampled_y

# 從目錄讀取，重新取樣並存檔
def do_resampling_andSave(ori_wavroot, save_resampled_wavroot='', target_sr=8000, time_domain_energyNorm=True):
    print('strat...')
    c = 0
    wavnum = len(os.listdir(ori_wavroot))
    for filename in sorted(os.listdir(ori_wavroot)):
        ori_wavpath = os.path.join(ori_wavroot, filename)
        reampled_y = resmple_wav(wav=ori_wavpath, target_sr=target_sr)
        # norm
        if time_domain_energyNorm:
            reampled_y = librosa.util.normalize(reampled_y)*0.8
        # save
        sf.write(os.path.join(save_resampled_wavroot, filename), reampled_y, target_sr, 'PCM_16')
        c = c + 1
        print(' [{}/{}]'.format(c, wavnum))

if __name__ == '__main__':

    for i in ['outside_test']:
        for j in ['speech','noisy','noise']:
            ori_wavroot =            '/media/hd03/sutsaiwei_data/data/mydata'+'/'+i+'/'+j
            save_resampled_wavroot = '/media/hd03/sutsaiwei_data/data/mydata2'+'/'+i+'/'+j
            target_sr = 8000
            print("ori_wavroot:",ori_wavroot)
            print("save_resampled_wavroot:",save_resampled_wavroot)
            # 路徑確認
            if not os.path.exists(save_resampled_wavroot):
                os.makedirs(save_resampled_wavroot)

            do_resampling_andSave(   
                        ori_wavroot=ori_wavroot, 
                        save_resampled_wavroot=save_resampled_wavroot, 
                        target_sr=target_sr)
            print('OK')
