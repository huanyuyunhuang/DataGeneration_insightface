from itertools import count
import os
import cv2
import sys
import dlib
import numpy as np
import matplotlib.pyplot as plt
from utils.SaveData import savedata
from utils.SignalGeneration import SignalGeneration
sys.path.append('..')

######################参数选择##########################
raw_video_path = r'E:\ff++\manipulated_sequences\Deepfakes\raw\videos\000_003.mp4'
c23_video_path = r'E:\ff++\manipulated_sequences\Deepfakes\c23\videos\000_003.mp4'
c40_video_path = r'E:\ff++\manipulated_sequences\Deepfakes\c40\videos\000_003.mp4'
raw_camera = cv2.VideoCapture(raw_video_path)
c23_camera = cv2.VideoCapture(c23_video_path)
c40_camera = cv2.VideoCapture(c40_video_path)
FrameCounter = 0 
Series_Group = np.zeros((28, int(Num_of_frame))) 
#######################################################
while raw_camera.isOpened():
    grabbed_r, raw_frame = raw_camera.read()
    grabbed_23, c23_frame = c23_camera.read()
    grabbed_40, c40_frame = c40_camera.read()
    if grabbed_r == False:
        break

