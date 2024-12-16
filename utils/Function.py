import cv2
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getROImask(landmark, frame):
    '''
    函数用于从脸上找到7个ROI（左右脸颊各两个、下巴、额头，鼻子），并返回各个ROI的mask。
    '''
    Plot = True                                                                 #画图标识符
    #以下定义各个ROI区域的Landmark序号
    #append的用法:可以向列表末尾添加元素
    ROI = []
    ROI.append([9,10,11,12,13,14,77,39,37,33,36,35,9])                              #左脸颊1
    ROI.append([14,15,16,2,3,4,52,77,14])                                         #左脸颊2
    ROI.append([4,5,6,7,8,0,24,23,22,21,20,61,58,59,53,56,55,52,4])                      #下巴
    ROI.append([20,19,18,32,31,30,83,61,20])                                     #右脸颊2
    ROI.append([30,29,28,27,26,25,93,91,87,90,89,83,30])                            #右脸颊1
    ROI.append([75,72,81,86])                                                       #鼻子
    ROI.append([43,48,49,51,50,102,103,104,105,101])                                 #眉毛（额头）
    ROI.append([35,41,40,42,39,37,33,36])                                          #左眼
    ROI.append([89,95,94,96,83,91,87,90])                                          #右眼
    
    ROI_coordinate = getFeaturePoints(ROI, landmark)                                #将ROI序号转换为坐标数组

    #计算鼻子区域RoI
    #由LandMark关键点定位鼻子四边形ROI的四点坐标，并更新替代原LandMark关键点坐标
    xmin, xmax, ymin, ymax = ROI_coordinate[5][0][0], ROI_coordinate[5][2][0], ROI_coordinate[5][1][1], ROI_coordinate[5][3][1]
    ROI_coordinate[5][0] = [xmin, ymin]
    ROI_coordinate[5][1] = [xmin, ymax]
    ROI_coordinate[5][2] = [xmax, ymax]
    ROI_coordinate[5][3] = [xmax, ymin]
    ROI_coordinate[5].append([xmin, ymin])

    #按先验计算额头区域
    left_eye = np.mean(np.array(ROI_coordinate[7]), axis=0)                         #左眼的平均坐标
    right_eye = np.mean(np.array(ROI_coordinate[8]), axis=0)                        #右眼的平均坐标
    eye_distance = np.linalg.norm(right_eye - left_eye)                             #眼睛之间的距离
    #temp=（左眉毛均值+右眉毛均值）/2-（左眼＋右眼）/2
    temp = (((np.mean(np.array(ROI_coordinate[6][0:5]), axis=0) + 
    np.mean(np.array(ROI_coordinate[6][5:10]), axis=0))/2 - (left_eye + right_eye)/2))
    temp = eye_distance/np.linalg.norm(temp)*0.6*temp
    Add_coordinate1 = list(ROI_coordinate[6][0] + temp)                             #额头左上角坐标
    Add_coordinate2 = list(ROI_coordinate[6][-1] + temp)                            #额头右上角坐标
    ROI_coordinate[6].append(Add_coordinate2)
    ROI_coordinate[6].append(Add_coordinate1)
    ROI_coordinate[6].append(ROI_coordinate[6][0])

    mask = ROI_mask(ROI_coordinate[0:7], frame.shape[0], frame.shape[1])            #分别计算ROI区域的mask

    #画出ROI区域
    if Plot == True:
        for i in range(len(mask)):
            cv2.polylines(frame, np.int32([ROI_coordinate[i]]), isClosed=True, color=1)
        cv2.imshow("Vedio",frame)
        cv2.waitKey(1)
    
    return mask, ROI_coordinate[0:7]


def ROI_mask(ROI_coordinate_list, w, h):
    '''
    函数用于对给定的多边形坐标生成Mask
    '''
    ROI_num = len(ROI_coordinate_list)
    mask = np.zeros((ROI_num, w, h), dtype="uint8")
    for i in range(ROI_num):
        ROI_coordinate = ROI_coordinate_list[i]
        cv2.polylines(mask[i], np.int32([ROI_coordinate]), isClosed=True, color=1)
        cv2.fillPoly(mask[i], np.int32([ROI_coordinate]), 1)
    return mask


def getFeaturePoints(ROI, lands):
    '''
    函数用于将LandMark索引转换为对应坐标
    '''
    ROI_coordinate = []
    temp_list = []
    for i in range(len(ROI)):
        processing_ROI = ROI[i]
        for j in range(len(processing_ROI)):
            temp_list.append([lands[ROI[i][j]][0], lands[ROI[i][j]][1]])
        ROI_coordinate.append(temp_list)
        temp_list = []
    return ROI_coordinate


def getTrackingPoints(face, NumOfPoints=40):
    '''
    函数用于均匀生成追踪点，face：检测到的脸部区域， NumberofPoints：生成追踪点的个数（平方）
    '''
    TrackingPoints = []
    FaceWidth = face[2]-face[0]+1                             #脸部宽、高
    FaceHeight = face[3]-face[1]+1
    dx = FaceWidth/(NumOfPoints + 1)                                             #x方向的打点间距
    dy = FaceHeight/(NumOfPoints + 1)                                            #y方向的打点间距
    Start_coordinate = [face[0]+dx, face[1]+dy]                              #起始坐标
    for i in range(NumOfPoints):
        for j in range(NumOfPoints):
            point = [Start_coordinate[0]+(i*dx), Start_coordinate[1]+(j*dy)]        
            TrackingPoints.append(point)
    TrackingPoints = np.float32(np.reshape(np.array(TrackingPoints), (NumOfPoints*NumOfPoints, 1, 2)))
    return TrackingPoints.tolist()

def TrackingDoubleOptical(old_frame, frame, FeaturePoints):
    '''
    函数用于双向光流跟踪，返回追踪成功特征点的FBerror和位移
    '''
    lk_params = dict(winSize = (15, 15),                                            #LK光流跟踪参数
                     maxLevel = 2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    old_frame_g = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)                       
    frame_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)                               #把上帧和这帧图像转为灰度图
                                                                                    #前向光流
    point_f, sucess_f, _ = cv2.calcOpticalFlowPyrLK(old_frame_g, frame_g, FeaturePoints, None, **lk_params)
                                                                                    #后向光流
    point_b, sucess_b, _ = cv2.calcOpticalFlowPyrLK(frame_g, old_frame_g, point_f, None, **lk_params)
                                                                                    #将前向和后向中追踪不成功的特征点去掉
    FeaturePoints, point_f, point_b = CheckTrackingSucess(FeaturePoints, point_f, point_b, sucess_f, sucess_b)
    FBError = CalculateFBError(FeaturePoints, point_b)                              #计算初始特征点和后向光流跟踪点之间的FBError
    Displacement = CalculateDisplacement(FeaturePoints, point_f)                    #计算初始特征点和前向光流跟踪点之间的位移
    return FBError, FeaturePoints, Displacement

def CheckTrackingSucess(FeaturePoints, point_f, point_b, sucess_f, sucess_b):
    '''
    函数用于将前向和后向中追踪不成功的特征点去掉，FeaturePoints：初始特征点，point_f：前向跟踪后的特征点
    point_b：后向跟踪后的特征点，sucess_f：前向跟踪成功标识符，sucess_b：后向跟踪成功标识符
    '''
    new_sucess = []
    new_FeaturePoints = []
    new_point_f = []
    new_point_b = []
    for i in range(sucess_f.shape[0]):                                              #前后向跟踪有一次不成功的均排除
        if sucess_f[i,0] == 1 and sucess_b[i,0] == 1:
            new_sucess.append(1)
        else:
            new_sucess.append(0)

    for i in range(len(new_sucess)):                                                #对跟踪成功的特征点进行更新
        if new_sucess[i] == 1:
            new_FeaturePoints.append(FeaturePoints[i])
            new_point_f.append(point_f[i])
            new_point_b.append(point_b[i])
    return np.array(new_FeaturePoints), np.array(new_point_f), np.array(new_point_b)


def CalculateFBError(point1, point2):
    '''
    函数用于计算FBerror
    '''
    FBError = []
    for i in range(point1.shape[0]):
        FBError.append(np.linalg.norm(point2[i,0] - point1[i,0]))                   #计算对应点之间的欧氏距离
    return FBError


def CalculateSmallerIndex(list_, percent):
    '''
    函数用于返回给定列表中较小的数的索引，list_:给定列表，percent：提取的百分比
    '''
    num = int(percent * len(list_))                                                 #计算需要提取多少个索引
                                                                                    #对列表从小到大排序
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[True])
    SmallIndex = list(N_large.index)[:num]
    biggest = N_large.score[N_large.index[num-1]]
    for i in range(num, len(list_)):                                                #提取最小的50%的索引
        if N_large.score[N_large.index[i]] == biggest:                              #如果有与第50%相同大小的数，一同放进列表里
            SmallIndex.append(N_large.index[i])
        else:
            break
    return SmallIndex

def CalculateDisplacement(FeaturePoints, point_f):
    '''
    函数用于计算前向光流的位移
    '''
    Displacement = []
    for i in range(FeaturePoints.shape[0]):
        displacement_x = point_f[i,0,0] - FeaturePoints[i,0,0]
        displacement_y = point_f[i,0,1] - FeaturePoints[i,0,1]
        Displacement.append([displacement_x, displacement_y])
    return Displacement
