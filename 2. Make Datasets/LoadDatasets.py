import matplotlib.pyplot as plt
import numpy as np
from keras.utils.io_utils import HDF5Matrix

ratio = 10
patch_n = 8
filename = 'Dataset_r%d_p%d_t1.hdf5' % (ratio, ratio*patch_n)

LR_set = HDF5Matrix(filename, 'LR')[:]
HR_set = HDF5Matrix(filename, 'HR')[:]
Dis_set = HDF5Matrix(filename, 'Dis')[:]
Subject_set = HDF5Matrix(filename, 'Subject')[:]
Position_set = HDF5Matrix(filename, 'Position')[:]
print(["Total Patches: ", len(LR_set)])

plt.gray()
for i in range(1750,len(LR_set)):
    plt.axis('equal')
    plt.subplot(2, 3, 1)
    plt.title("Subject: %d,  X: %d,  Y: %d" % (Subject_set[i], Position_set[i, 0], Position_set[i, 1]))
    plt.imshow(LR_set[i,:,:,0], vmin=0, vmax=1, origin='lower')
    plt.subplot(2, 3, 3)
    plt.imshow(HR_set[i, :, :,0], vmin=0, vmax=1, origin='lower')
    plt.subplot(2, 3, 4)
    plt.xlim(-1, patch_n+1)
    plt.ylim(-1, patch_n+1)

    # Displacement Plot
    arrow_scale = 10
    for j in range(3):
        plt.subplot(2, 3, 4 + j)
        plt.xlim(-1, patch_n+1)
        plt.ylim(-1, patch_n+1)
        for y in range(patch_n+1):
            for x in range(patch_n+1):
                # if x==0 or y == 0 or x==patch_n or y==patch_n:
                plt.arrow(x, y, Dis_set[i, y, x, j*2]*ratio*arrow_scale, Dis_set[i,y,x,j*2+1]*ratio*arrow_scale)
                plt.scatter(x, y, s=5)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
