import numpy as np
import matplotlib.pyplot as plt
import math
import os

ratio = 8
N = 3
img = plt.imread("Subject.jpg")
[height, width, channel] = img.shape
os.mkdir("LR_%d/%d" % (ratio, N)) # ratio만큼 다운 스케일 후 N 만큼의 subpatch 따라서 HR에서는 ratio * N의 윈도우 크기를 가지고 있음.
for i in range(math.floor(height/N)):
    for j in range(math.floor(width/N)):
        sub_img = img[i*N:(i+1)*N, j*N:(j+1)*N, :]
        plt.imsave('LR_%d\%d\%d.jpg' % (ratio, N, i*N+j+1), sub_img)
