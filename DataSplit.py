import os
import numpy as np
import shutil
import matplotlib.pyplot as plt

#DataSplit：将原始视频处理后的数据按64长度、1s滑动窗口分割数据

def Normalization(arr):
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        temp = arr[i]
        result[i] = (temp - min(temp))/(max(temp) - min(temp))
    return result


video_type = 'Faceswap'
video_quality = 'c40'
DataFolder_path = os.path.join('../c40Data', video_type, video_quality)
Save_root = os.path.join('../Data', video_type, video_quality)
WindowLength = 64
train_save_index = 0
val_save_index = 0
test_save_index = 0
use50_flag=True

File = os.listdir(DataFolder_path)
for file in File:
    print('processing:', video_quality + '_' + video_type + '_' + file)
    file_path = os.path.join(DataFolder_path, file)
    video_log_path = os.path.join(file_path, 'video_log.txt')
    Framerate_path = os.path.join(file_path, 'Framerate.txt')
    Flag_path = os.path.join(file_path, 'Flag.txt')

    mean = np.load(os.path.join(file_path, 'mean.npy'))
    median = np.load(os.path.join(file_path, 'median.npy'))
    SlideLength = Framerate = np.loadtxt(Framerate_path, dtype=np.int32)

    meandata = Normalization(mean)#归一化
    mediandata = Normalization(median)  # 归一化
    FrameNum = meandata.shape[1]
    ClipNum = np.floor((FrameNum - WindowLength)/SlideLength)

    if int(file) <= 900:
        for i in range(int(ClipNum)):
            begin_idx = int(np.floor(Framerate * i))
            end_idx = begin_idx + WindowLength
            if use50_flag==True:
                saving_data = np.concatenate((meandata[0:14,begin_idx : end_idx],mediandata[0:14,begin_idx : end_idx]),0)
            else:
                saving_data = np.concatenate((meandata[14:28, begin_idx: end_idx], mediandata[14:28, begin_idx: end_idx]),0)
            if int(file) <= 800:
                train_save_index += 1
                Save_path = os.path.join(Save_root, 'train', str(train_save_index))
            else:
                val_save_index += 1
                Save_path = os.path.join(Save_root, 'val', str(val_save_index))

            exist = os.path.exists(Save_path)  
            if exist:
                raise ValueError('Folder already exists')
            else:
                os.mkdir(Save_path)

            #新建文件
            new_video_log_path = os.path.join(Save_path, 'video_log.txt')
            new_Framerate_path = os.path.join(Save_path, 'Framerate.txt')
            new_Flag_path = os.path.join(Save_path, 'Flag.txt')

            #shutil.copyfile：参数一内容复制到参数二
            shutil.copyfile(video_log_path, new_video_log_path)
            shutil.copyfile(Framerate_path, new_Framerate_path)
            shutil.copyfile(Flag_path, new_Flag_path)

            data_path = os.path.join(Save_path, 'data.npy')
            np.save(data_path, saving_data)
    #测试集一个data存在一个文件夹
    else:
        for i in range(int(ClipNum)):
            begin_idx = int(np.floor(Framerate * i))
            end_idx = begin_idx + WindowLength
            if use50_flag == True:
                saving_data = np.concatenate((meandata[0:14, begin_idx: end_idx], mediandata[0:14, begin_idx: end_idx]),0)
            else:
                saving_data = np.concatenate((meandata[14:28, begin_idx: end_idx], mediandata[14:28, begin_idx: end_idx]),0)
            test_save_index += 1
            Save_path = os.path.join(Save_root, 'test', str(test_save_index))
            exist = os.path.exists(Save_path)
            if exist:
                raise ValueError('Folder already exists')
            else:
                os.mkdir(Save_path)

            new_video_log_path = os.path.join(Save_path, 'video_log.txt')
            new_Framerate_path = os.path.join(Save_path, 'Framerate.txt')
            new_Flag_path = os.path.join(Save_path, 'Flag.txt')

            shutil.copyfile(video_log_path, new_video_log_path)
            shutil.copyfile(Framerate_path, new_Framerate_path)
            shutil.copyfile(Flag_path, new_Flag_path)

            data_path = os.path.join(Save_path, 'data.npy')
            np.save(data_path, saving_data)
    #原方案
    # else:
    #     test_save_index += 1
    #     Save_path = os.path.join(Save_root, 'test', str(test_save_index))
    #     exist = os.path.exists(Save_path)
    #     if exist:
    #         raise ValueError('Folder already exists')
    #     else:
    #         os.mkdir(Save_path)
    #
    #     new_video_log_path = os.path.join(Save_path, 'video_log.txt')
    #     new_Framerate_path = os.path.join(Save_path, 'Framerate.txt')
    #     new_Flag_path = os.path.join(Save_path, 'Flag.txt')
    #
    #     shutil.copyfile(video_log_path, new_video_log_path)
    #     shutil.copyfile(Framerate_path, new_Framerate_path)
    #     shutil.copyfile(Flag_path, new_Flag_path)
    #
    #     for i in range(int(ClipNum)):
    #         begin_idx = int(np.floor(Framerate * i))
    #         end_idx = begin_idx + WindowLength
    #         if use50_flag == True:
    #             saving_data = np.concatenate((meandata[0:14, begin_idx: end_idx], mediandata[0:14, begin_idx: end_idx]),0)
    #         else:
    #             saving_data = np.concatenate((meandata[14:28, begin_idx: end_idx], mediandata[14:28, begin_idx: end_idx]),0)
    #
    #         data_path = os.path.join(Save_path, 'data' + str(i) + '.npy')
    #         np.save(data_path, saving_data)