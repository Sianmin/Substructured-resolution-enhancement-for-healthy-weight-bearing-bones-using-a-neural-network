import numpy as np
import matplotlib.pyplot as plt
import math
import miscell as m

ratio = 10
TEST = 1

img = m.rgb2gray(plt.imread("Subject_%d.jpg" % TEST))
LR_img = m.rgb2gray(plt.imread("Subject_%d_LR_%d.jpg" % (TEST, ratio)))

[height, width] = img.shape
plt.gray()

m.mkdir("HR")
m.mkdir("LR")
for N in range(30, 31): #각 윈도우 탐색
    n = int(N/ratio)

    m.mkdir("HR\%d" % N)

    m.mkdir("LR\Down%d" % ratio)
    m.mkdir("LR\Down%d\%d" % (ratio, n))

    for i in range(math.floor(height/N)):
        for j in range(math.floor(width/N)):
            sub_img = img[i*N:(i+1)*N, j*N:(j+1)*N]
            sub_LR_img = LR_img[i*n:(i+1)*n, j*n:(j+1)*n]
            plt.imsave('HR\%d\%d.jpg' % (N, i*math.floor(width/N)+j+1), sub_img)
            if N % 10 == 0:
                plt.imsave('LR\Down%d\%d\%d.jpg'%(ratio, n, i*math.floor(width/N)+j+1), sub_LR_img)




# for ratio in range(2, 11):
#     [height, width, channel] = img.shape
#     print(img.shape)
#     LR_img = np.zeros((math.floor(height/ratio), math.floor(width/ratio)))
#     for i in range(math.floor(height/ratio)):
#         for j in range(math.floor(width/ratio)):
#             sub_img = img[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio, :]
#             LR_img[i, j] = np.sum(sub_img)/ (ratio**2)
#     plt.imsave('Subject_LR_%d.jpg' % ratio, LR_img.astype(np.uint8), cmap='gray')
