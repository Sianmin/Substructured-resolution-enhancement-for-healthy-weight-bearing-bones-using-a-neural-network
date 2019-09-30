# 전체 모델간 비교를 하기 위한 파일
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr
from scipy.misc import imresize
import warnings
import cv2
warnings.filterwarnings("ignore")

[height, width] = [2080, 1883]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
def Compare(name, ref_img, target_img, save=True):
    print("---%s---"%name)
    print("SSIM:\t%f" % ssim(ref_img, target_img))
    print("MSE:\t%f" % mse(ref_img, target_img))
    if save:
        plt.gray()
        plt.imsave("%s.png"%name, target_img)
def Get_Boundary_Variation(img, size, order = 'lower'):
    if order == 'lower':
        img = np.flip(img, axis = 0)
    [NX, NY] = [math.floor(img.shape[1] / size), math.floor(img.shape[0] / size)]

    # img = (1 - img)
    # img = np.asarray([img, img, img]).transpose((1, 2, 0))
    BV = 0
    n = 0
    for i in range(0, NY):
        for j in range(1, NX):
            BV += np.sum(np.abs(img[i*size:(i+1)*size , j * size-1] - img[i*size:(i+1)*size, j * size]))
            n += size
            # img[i * size: (i + 1) * size, j * size - 2: j*size+2, :] = 1
            # img[i * size: (i + 1) * size, j * size - 2: j*size+2, 1:3] = 0
    for i in range(0, NX):
        for j in range(1, NY):
            BV += np.sum(np.abs(img[j * size-1, i*size:(i+1)*size] - img[j * size, i*size:(i+1)*size]))
            n += size
            # img[j * size - 2: j*size+2, i * size:(i + 1) * size, :] = 1
            # img[j * size - 2 : j*size+2, i * size:(i + 1) * size, 0:2] = 0
    # plt.gray()
    # plt.imsave("PBV.png", img, origin='lower')
    # plt.imshow(img, origin='lower')
    # plt.show()

    return BV/n

if __name__ == '__main__':
    [subject, ratio, patch_n] = [1, 10, 80]
    [NX, NY] = [math.floor(width/ratio), math.floor(height/ratio)]

    # 영상 불러오기
    LR_img = rgb2gray(plt.imread('../0. Datas and Preprocessing/IMAGE/r%d/IMG_r%d_s%d.png' % (ratio, ratio, subject)))
    ref_img = rgb2gray(plt.imread('../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s%d.png' % subject))
    print(["원본 BV : ", Get_Boundary_Variation(ref_img, patch_n)])

    # Jungjinnet
    comp_path = "Completed/JJNet-09-04-13-31/ 5-predict_subject%d.png" % subject
    comp_img = rgb2gray(plt.imread("../3. Neural Network/Models/" + comp_path))
    Compare("JJNET", ref_img, comp_img)
    print(["BV : ", Get_Boundary_Variation(comp_img, patch_n)])

    # SRGAN
    comp_path = "Completed/SRGAN-09-11-12-47/05-predict_subject%d.png" % subject
    comp_img = rgb2gray(plt.imread("../3. Neural Network/Models/" + comp_path))
    Compare("SRGAN", ref_img, comp_img)
    print(["BV : ", Get_Boundary_Variation(comp_img, patch_n)])

    # Feasibility_GAN
    comp_path = "Completed/Feasibility/Feasibility%d_f.png" % subject
    comp_img = rgb2gray(plt.imread("../3. Neural Network/Models/" + comp_path))
    Compare("Feasibility", ref_img, comp_img)
    print(["BV : ", Get_Boundary_Variation(comp_img, patch_n)])

    # Feasibility_CNN
    comp_path = "Completed/Feasibility/CNN_Feasibility%d.png" % subject
    comp_img = rgb2gray(plt.imread("../3. Neural Network/Models/" + comp_path))
    Compare("Feasibility", ref_img, comp_img)
    print(["BV : ", Get_Boundary_Variation(comp_img, patch_n)])

    # Nearest
    inter_img = imresize(LR_img, (height, width), 'nearest')/255
    Compare('nearest', ref_img, inter_img)
    print(["BV : ", Get_Boundary_Variation(inter_img, patch_n)])

    # Bilinear
    inter_img = imresize(LR_img, (height, width), 'bilinear')/255
    Compare('bilinear', ref_img, inter_img)
    print(["BV : ", Get_Boundary_Variation(inter_img, patch_n)])

    #Bicubic
    inter_img = imresize(LR_img, (height, width), 'bicubic')/255
    Compare('bicubic', ref_img, inter_img)
    print(["BV : ", Get_Boundary_Variation(inter_img, patch_n)])

    # Lanczos
    inter_img = imresize(LR_img, (height, width), 'lanczos')/255
    Compare('lanczos', ref_img, inter_img)
    print(["BV : ", Get_Boundary_Variation(inter_img, patch_n)])