import os
import sys
import numpy as np
sys.path.append('..')

def savedata(mean, median, mode, depth, Flag, Framerate, index, name, Target_root):
    Save_Folder_Path = os.path.join(Target_root, str(index))
    exist = os.path.exists(Save_Folder_Path)                        #检测文件夹是否存在，若不存在则建立
    if exist:
        raise ValueError('Folder already exists')
    else:
        os.mkdir(Save_Folder_Path)

    mean_Save_Path = os.path.join(Save_Folder_Path, 'mean.npy')
    median_Save_Path = os.path.join(Save_Folder_Path, 'median.npy')
    mode_Save_Path = os.path.join(Save_Folder_Path, 'mode.npy')
    depth_Save_Path = os.path.join(Save_Folder_Path, 'depth.npy')
    #Color_Save_Path = os.path.join(Save_Folder_Path, 'Color.npy')
    Flag_Save_Path = os.path.join(Save_Folder_Path, 'Flag.txt')
    Framerate_Save_Path = os.path.join(Save_Folder_Path, 'Framerate.txt')
    logfile_Save_Path = os.path.join(Save_Folder_Path, 'video_log.txt')

    # added by 2022-01-28
    np.save(mean_Save_Path, mean)
    np.save(median_Save_Path, median)
    np.save(mode_Save_Path, mode)
    np.save(depth_Save_Path,depth)
    #np.save(Color_Save_Path, Color)

    f = open(Flag_Save_Path, 'w')
    f.write(Flag)
    f.close()
    f = open(Framerate_Save_Path, 'w')
    f.write(str(Framerate))
    f.close()
    f = open(logfile_Save_Path, 'w')
    f.write(str(name))
    f.close()