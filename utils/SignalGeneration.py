import cv2
import numpy as np
import utils.Function as F
import matplotlib.pyplot as plt

def SignalGeneration(mean, median, mode, depth, old_frame, frame, old_face, face, old_ROI_coordinate, landmark, depth_landmark,index, miss):
    mask, ROI_coordinate = F.getROImask(landmark, frame)                    #获取各个ROi的mask 
    #Color = getROIPixelAndValue(Color, mask, frame, index)                  #计算各个ROI的RGB空间均值
    if index != 0:                                                          #如果为第一帧，则设各个ROI轨迹起始值为0
                                                                            #否则，则计算各个ROI的位移均值、中值
        old_face, old_frame, mean, median, mode, depth, miss = (getROITrajectory(mean, median, mode, depth, depth_landmark, old_ROI_coordinate,
        old_face, face, old_frame, frame, miss, index))
    else:
        depth[0,index]=np.mean(depth_landmark)
    return old_face, old_frame, ROI_coordinate, mean, median, mode, depth, miss

    
def getROIPixelAndValue(Color, mask, img, index):
    '''
    函数用于统计一帧中各个ROI区域的像素数目以及RGB3个通道的像素和并形成一路信号
    '''
    Signal_temp = np.zeros((mask.shape[0], img.shape[2]))                   #记录7个区域RGB平均值（7×3）
    pixel_count = np.zeros(mask.shape[0])                                   #记录7个区域的像素个数（7）

    #首先统计各个ROI的像素个数
    for i in range(mask.shape[0]):
        processing_mask = mask[i]                                           
        count = np.where(processing_mask == 1)[0].shape[0]                  #计算mask内像素个数
        pixel_count[i] = count

    #然后统计各个ROI的RGB通道像素总值
    for i in range(mask.shape[0]):
        processing_mask = mask[i]
        temp_img = cv2.bitwise_and(img, img, mask=processing_mask)
        Signal_temp[i] = np.sum(temp_img, axis=(0,1))

    #求平均并写入COLOR的一帧
    for i in range(Signal_temp.shape[0]):
        Color[i*3:(i+1)*3, index] = Signal_temp[i] / pixel_count[i]
    return Color



def getROITrajectory(mean, median, mode, depth, depth_landmark, old_ROI_coordinate, old_face, face, old_frame, frame, miss, index, NumOfPoints=40):
    '''
    函数用于计算前后向光流，并统计FBError最小的50%的点的中值和均值作为ROI的位移值
    '''
    FeaturePoints = F.getTrackingPoints(old_face, NumOfPoints=NumOfPoints)  #均匀生成追踪的特征点
    for i in range(len(old_ROI_coordinate)):
        for j in range(len(old_ROI_coordinate[i]) - 1):
            FeaturePoints.append([old_ROI_coordinate[i][j]])
    FeaturePoints = np.array(FeaturePoints, dtype=np.float32)
            
                                                                            #计算两帧之间的前后向光流，返回跟踪成功的特征点，相应的FBError和位移
    FBerror, FeaturePoints, Displacement = F.TrackingDoubleOptical(old_frame, frame, FeaturePoints)

    if len(FeaturePoints) < (0.2 * NumOfPoints * NumOfPoints):
        miss.append(index)
        return face, frame, mean, median, mode, miss

    #对每个ROI做处理
    for i in range(len(old_ROI_coordinate)):
        displacement = []
        processing_FBError = []
        processing_displacement = []
                                                                            #处理的ROI的边界多边形
        processing_cnt = np.reshape(np.array(old_ROI_coordinate[i], dtype=np.int32), (len(old_ROI_coordinate[i]), 1, 2))    ####旧的RoI
        for j in range(FeaturePoints.shape[0]):
                                                                            #判断特征点是否在ROI边界多面形内
            dist = cv2.pointPolygonTest(processing_cnt, (FeaturePoints[j,0,0], FeaturePoints[j,0,1]), False)
            if dist >= 0:                                                   #如果在，则将该特征点对应的位移和FBError提取出来
                processing_FBError.append(FBerror[j])
                processing_displacement.append(Displacement[j])  
        if len(processing_FBError) == 0:
            miss.append(index)
            return face, frame, mean, median, mode, miss
           
        small_FB = F.CalculateSmallerIndex(processing_FBError, 0.5)         #计算ROI内特征点FBError较小的50%的索引
        small_FB.sort()
        for indexs in small_FB:
            displacement.append(processing_displacement[indexs])            #将FBError较小的50%的特征点的位移提出
        #为啥精简版本和原始版本都计算？
        displacement = np.array(displacement)
        mean_value = np.mean(displacement, axis=0)                          #计算均值、中值、最小值、最大值、众数
        median_value = np.median(displacement, axis=0)
        # why? 这里干了啥
        hist_x, bin_x = np.histogram(displacement[:,0], bins=10,density=False)
        hist_y, bin_y = np.histogram(displacement[:,1], bins=10,density=False)
        #最值
        x_min, x_max = bin_x[np.argmax(hist_x)], bin_x[np.argmax(hist_x) + 1]
        y_min, y_max = bin_y[np.argmax(hist_y)], bin_y[np.argmax(hist_y) + 1]
        x_list, y_list = [], []
        for k in range(displacement.shape[0]):
            if displacement[k,0] <= x_max and displacement[k,0] >= x_min:
                x_list.append(displacement[k,0])
            if displacement[k,1] <= y_max and displacement[k,1] >= y_min:
                y_list.append(displacement[k,1])
        #众数
        mode_value_x = np.mean(x_list)
        mode_value_y = np.mean(y_list)

        displacement_100 = np.array(processing_displacement)
        mean_value_100 = np.mean(displacement_100, axis=0)                          #计算均值、中值、最小值、最大值、众数
        median_value_100 = np.median(displacement_100, axis=0)
        #why? 这里干了啥
        hist_x_100, bin_x_100 = np.histogram(displacement_100[:,0], bins=10,density=False)
        hist_y_100, bin_y_100 = np.histogram(displacement_100[:,1], bins=10,density=False)
        x_min_100, x_max_100 = bin_x_100[np.argmax(hist_x_100)], bin_x_100[np.argmax(hist_x_100) + 1]
        y_min_100, y_max_100 = bin_y_100[np.argmax(hist_y_100)], bin_y_100[np.argmax(hist_y_100) + 1]
        x_list_100, y_list_100 = [], []
        for k in range(displacement_100.shape[0]):
            if displacement_100[k,0] <= x_max_100 and displacement_100[k,0] >= x_min_100:
                x_list_100.append(displacement_100[k,0])
            if displacement_100[k,1] <= y_max_100 and displacement_100[k,1] >= y_min_100:
                y_list_100.append(displacement_100[k,1])
        mode_value_x_100 = np.mean(x_list_100)
        mode_value_y_100 = np.mean(y_list_100)
        #几个数组的含义？
        mean[i, index] =  mean_value[0] + mean[i, index-1]     #写入轨迹数组
        mean[i+7, index] = mean_value[1] + mean[i+7, index-1]
        median[i, index] =  median_value[0] + median[i, index-1]
        median[i+7, index] = median_value[1] + median[i+7, index-1]
        mode[i, index] =  mode_value_x + mode[i, index-1]
        mode[i+7, index] = mode_value_y + mode[i+7, index-1]

        mean[i+14, index] =  mean_value_100[0] + mean[i+14, index-1]     #写入轨迹数组
        mean[i+21, index] = mean_value_100[1] + mean[i+21, index-1]
        median[i+14, index] =  median_value_100[0] + median[i+14, index-1]
        median[i+21, index] = median_value_100[1] + median[i+21, index-1]
        mode[i+14, index] =  mode_value_x_100 + mode[i+14, index-1]
        mode[i+21, index] = mode_value_y_100 + mode[i+21, index-1]

        depth[0,index] = np.mean(depth_landmark)

    return face, frame, mean, median, mode, depth, miss



    
