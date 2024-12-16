import cv2
import os
import insightface
from sklearn import preprocessing
import numpy as np

print("Package Imported")
class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
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
        FACEs = self.model.get(image)
        for FACE in FACEs:
            # 获取人脸属性
            #shape屬性的含义
            face = np.array(FACE.bbox).astype(np.int32).tolist()
            landmarks = np.array(FACE.landmark_3d_68).astype(np.int32)
            landmarks=landmarks[:,[0,1]]
            # result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            # result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            # result["landmark_3d_68"] = np.array(face.landmark_3d_68).astype(np.int32).tolist()
            # result["landmark_2d_106"] = np.array(face.landmark_2d_106).astype(np.int32).tolist()
        return face,landmarks

# 绘制直线和关键点
def draw_line(img, point1, point2):
    cv2.line(img, pt1=(point1[0], point1[1]), pt2=(point2[0], point2[1]),
             color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
#读取视频并播放
camera=cv2.VideoCapture("./001.mp4")
face_recognitio = FaceRecognition()
MisIndex=[]
FrameCounter = 0
while camera.isOpened():
    grabbed,frame=camera.read()
    if grabbed == False:  # 读终止
        break
    FrameCounter += 1
    print('Miss frame:{}'.format(MisIndex))
    #需要人脸矩形边框---矩形类rectangle---[(xx,xx) (xx,xx)](左上点坐标、宽度、高度)
    face, landmarks = face_recognitio.detect(frame)
    if len(face) == 0:
        MisIndex.append(FrameCounter - 1)
        continue
   #方形
   #shape[0]--左上角x shape[1]--左上角y shape[2]--右下角x shape[3]--右下角y
   #rectangle--[(shape[0],shape[1]) (...)]
    cv2.rectangle(frame, (face[0], face[1]), (face[2],face[3]), (255, 0, 0), 2)
    for i, point in enumerate(landmarks):
        x = point[0]
        y = point[1]
        cv2.circle(frame, (x, y), 1, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(img, str(i + 1), (x, y), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        #             fontScale=0.3, color=(255, 255, 0))
        # 连接关键点
        if i + 1 < 17:  # face
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 17 < i + 1 < 22:  # eyebrow1
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 22 < i + 1 < 27:  # eyebrow2
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 27 < i + 1 < 31:  # nose
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 31 < i + 1 < 36:  # nostril
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 36 < i + 1 < 42:  # eye1
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 42 < i + 1 < 48:  # eye2
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 48 < i + 1 < 60:  # lips
            draw_line(frame, landmarks[i], landmarks[i+1])
        elif 60 < i + 1 < 68:  # teeth
            draw_line(frame, landmarks[i], landmarks[i+1])

    cv2.imshow("Vedio",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):#按下q后if成立，&xFF取低8位
        break



