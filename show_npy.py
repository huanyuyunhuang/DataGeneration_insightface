import numpy as np
import cv2
import matplotlib.pyplot as plt

#可视化轨迹曲线
image1 = np.load('D:/Code_bishe/Data/Faceswap/c40/test/1/data0.npy')

X=64*np.linspace(0,1,64)
plt.figure(1)
for i in range(0,28):
    plt.plot(X, image1[i, :])
plt.show()
