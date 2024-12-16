from itertools import count
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.SaveData import savedata
from utils.SignalGeneration import SignalGeneration
import insightface
from sklearn import preprocessing
import torch
sys.path.append('..')
##########################检测模型############################
class FaceRecognition:
    def __init__(self, gpu_id=1, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()

    # 检测人脸
    def detect(self, image):
        tFACEs = self.model.get(image)
        if len(tFACEs)>0:#取第一个人脸
            face = np.array(tFACEs[0].bbox).astype(np.int32).tolist()
            landmark = np.array(tFACEs[0].landmark_2d_106).astype(np.int32)  # 提取二维数据
            # landmark=landmark[:,[0,1]]
            depth_landmark = np.array(tFACEs[0].landmark_3d_68)  # 提取三维数据
            depth_landmark = depth_landmark[:, 2]
        else:
            face=[]
            landmark=[]
            depth_landmark=[]
        return face,landmark,depth_landmark

######################参数选择##########################
Flag = 'NeuralTextures'                                                                #视频真假标志符
video_type = 'c40'                                                                     #视频类型
logFile = r'../c40Data/' + Flag + r'/logfile_' + Flag + '_' + video_type + '.txt'      #日志文件路径
Target_root = r'../c40Data/' + Flag + '/' + video_type                                 #存储路径
face_recognitio = FaceRecognition()                                                    #检测器初始化
#######################################################
if Flag == 'Real':
    Video_Folder_root = r'D:\ff++\original_sequences\youtube\\' + video_type + r'\videos'           #真视频地址
else:
    Video_Folder_root = r'D:\ff++\manipulated_sequences\\' + Flag + '\\'  + video_type + r'\videos'  #假视频地址

video_list = os.listdir(Video_Folder_root)          #获取文件列表
video_count = 0                               #进度
count=video_count
print(torch.cuda.get_device_name(0))
time=0
#######################################################
for name in video_list:                             #遍历
    if time<count:                                  #用于跳过已处理项，进度读取
        time=time+1
        print("The {} is Dealed.".format(name))
        continue
    video_count += 1
    video_path = os.path.join(Video_Folder_root, name)              #组成完整地址，具体到某个视频文件夹
    video_name = video_type + '_' + Flag + '_' + str(video_count)
    print('processing path:',video_path)
    print('processing video:', video_name)

    camera = cv2.VideoCapture(video_path)
    Framerate = int(camera.get(cv2.CAP_PROP_FPS))                                   #获取视频帧率---Framerate
    Num_of_frame = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))                        #获取视频帧总数---Num_of_frame
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))                               #获取帧宽度---width  等于说是图像宽高
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))                             #获取帧高度---height
    print("FrameRate:{}, FrameNum:{}, Frame_W:{}, Frame_H:{}".format(Framerate, Num_of_frame, width, height))

    FrameCounter = 0                                                                #帧计数器
    MisIndex = []                                                                   #未捕捉到的帧索引
    mean_Series_Group = np.zeros((28, int(Num_of_frame)))                     #初始化记录轨迹数组（7个区域的x和y的均值）
    median_Series_Group = np.zeros((28, int(Num_of_frame)))                   #初始化记录轨迹数组（7个区域的x和y的中值）
    mode_Series_Group = np.zeros((28, int(Num_of_frame)))                     #初始化记录轨迹数组（7个区域的x和y的众数）
    depth_Series_Group = np.zeros((1, int(Num_of_frame)))                    #初始化深度信息数组，每一帧取64个数据的均值

    while camera.isOpened():
        grabbed, frame = camera.read()

        if grabbed == False:                                                   #读终止
            break

        FrameCounter += 1                                                      #可读，则帧数加1
        print('num:{}   processing frame:{}   Miss frame:{}'.format(video_count,FrameCounter,MisIndex))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, landmark, depth_landmark = face_recognitio.detect(frame)                        #检测人脸
        # for i, point in enumerate(landmark):
        #     x = point[0]
        #     y = point[1]
        #     cv2.circle(frame, (x, y), 1, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        #     cv2.putText(frame, str(i + 1), (x, y), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        #             fontScale=0.3, color=(255, 255, 0))
        # cv2.imshow("Vedio",frame)
        # cv2.waitKey(1)
        if len(MisIndex) > 10:                                                      #光流跟踪失败的帧数大于10，放弃处理该帧
            break
        if len(face) == 0:
            MisIndex.append(FrameCounter - 1)
            continue
        else:
            face = face

        index = FrameCounter - 1
        if index == 0:                                                              #如果为第一帧，则old_face和old_frame为当前脸和当前帧
            old_frame = frame
            old_face = face
            old_coor = None
        old_face, old_frame, old_coor, mean_Series_Group, median_Series_Group, mode_Series_Group,depth_Series_Group,MisIndex = (SignalGeneration(mean_Series_Group,
        median_Series_Group, mode_Series_Group, depth_Series_Group, old_frame, frame, old_face, face, old_coor, landmark, depth_landmark, index, MisIndex))     #循环更新轨迹和颜色的记录数组
    camera.release()
#######################################################
    if len(MisIndex) < 10:
        if len(MisIndex) != 0:
            new_mean= np.zeros((28, int(Num_of_frame) - len(MisIndex)))
            new_median= np.zeros((28, int(Num_of_frame) - len(MisIndex)))
            new_mode= np.zeros((28, int(Num_of_frame) - len(MisIndex)))
            new_depth= np.zeros((1, int(Num_of_frame) - len(MisIndex)))
            n = 0
            for l in range(int(Num_of_frame)):
                if l not in MisIndex:
                    new_mean[:,n] = mean_Series_Group[:,l]
                    new_median[:,n] = median_Series_Group[:,l]
                    new_mode[:,n] = mode_Series_Group[:,l]
                    new_depth[:,n]=depth_Series_Group[:,l]
                    n += 1
            savedata(new_mean, new_median, new_mode, new_depth, Flag, Framerate, video_count, video_name, Target_root)
            #savedata(new_mean, new_median, new_mode, Flag, Framerate, count, video_name, Target_root)
        else:
            savedata(mean_Series_Group, median_Series_Group, mode_Series_Group, depth_Series_Group, Flag, Framerate, video_count, video_name, Target_root)
            #savedata(mean_Series_Group, median_Series_Group, mode_Series_Group, Flag, Framerate, count, video_name, Target_root)
        logfile = video_name + ': sucessfully processed '
    else:
        logfile = video_name + ': miscatch more than 10 frames'

    with open(logFile, "a") as f:
        f.write(logfile)
        f.write('\n')
