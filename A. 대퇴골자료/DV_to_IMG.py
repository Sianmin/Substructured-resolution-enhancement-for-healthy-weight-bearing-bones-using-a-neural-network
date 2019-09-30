import numpy as np
import matplotlib.pyplot as plt
import math
import time


for TEST in range (1,12):
    file = open("CASE %d\CASE%d_XDEN3_HR.DAT" % (TEST, TEST), 'r')
    Total = np.zeros((2080, 1883))
    while 1:
        temp = (file.readline().split())
        if not temp: break
        [temp1, temp2, temp3] = [eval(temp[0]), eval(temp[1]), eval(temp[2])]
        Total[temp2 - 1, temp1 - 1] = temp[2]
        #Total[2078-(temp2-1), temp1-1] = temp[2]
    print(Total.shape)
    file.close()
    plt.imsave('Subject_%d.jpg' %TEST, Total, cmap = 'gray', origin = 'lower')
    img = plt.imread("Subject_%d.jpg"%TEST)
    print(img.shape)
# Total = np.zeros((11*156, 11*156))
# print(HR.shape)
#
# for t in range(11):
#     print(t)
#     for i in range(120):
#         #print([156*math.floor(i/11), 156*(i%11)])
#         Total[156*((i+1)%11):156*((i+1)%11)+156, 156*math.floor(i/11):156*math.floor(i/11)+156] = (HR[t, i, :, :])

#for i in range(60):
 #   for j in range(60):
  #      Total[156*(i-1)+1:156*i, 156*(j-1)+1:156*j] = HR[1, (i-1)*120 + j , :, :]
#plt.imshow(Total)
#plt.show(

    #print(i)
    #plt.pause(3)
    #plt.close()