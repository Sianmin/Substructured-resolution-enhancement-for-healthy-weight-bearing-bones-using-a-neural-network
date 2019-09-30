import numpy as np
import matplotlib.pyplot as plt
import math
import os

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

TEST = 1
ratio = 10

img = rgb2gray(plt.imread("Subject_%d.jpg" % TEST))
for ratio in range(2, 11):
    [height, width] = img.shape
    print(img.shape)
    LR_img = np.zeros((math.floor(height/ratio), math.floor(width/ratio)))
    for i in range(math.floor(height/ratio)):
        for j in range(math.floor(width/ratio)):
            sub_img = img[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
            LR_img[i, j] = np.sum(sub_img)/ (ratio**2)
    plt.imsave('Subject_%d_LR_%d.jpg' % (TEST, ratio), LR_img.astype(np.uint8), cmap='gray')
