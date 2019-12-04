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
def Compare(name, ref_img, target_img, save=False):
    print("---%s---"%name)
    print("SSIM:\t%f" % ssim(ref_img, target_img, data_range = 1))
    print("MSE:\t%f" % mse(ref_img, target_img))
    print("PSNR:\t%f" % psnr(ref_img, target_img, data_range=1))
    if save:
        plt.gray()
        plt.imsave("%s.png"%name, target_img)
def Get_Boundary_Variation(img, mask, order='higher'):
    if order == 'lower':
        img = np.flip(img, axis = 0)
    height, width = img.shape[0], img.shape[1]
    BV, n = 0, 0

    # import cv2
    # sobelx = cv2.Sobel(img, cv2.CV_64F,1, 0)
    # sobely = cv2.Sobel(img, cv2.CV_64F,0, 1)
    # magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    # plt.gray()
    # plt.subplot(121)
    # plt.imshow(magnitude)
    # plt.subplot(122)
    # plt.imshow(mask)
    # plt.show()
    for i in range(height-1):
        for j in range(width-1):
            if mask[i, j] == 1:
                if mask[i+1, j] != 1:
                    BV += abs(img[i,j] - img[i+1,j])
                if mask[i, j+1] != 1:
                    BV += abs(img[i, j] - img[i , j+1])
                n += 1

    return BV/n

if __name__ == '__main__':
    [subject, ratio, patch_n] = [1, 10, 8]
    [NX, NY] = [math.floor(width/ratio), math.floor(height/ratio)]
    rp = ratio*patch_n

    ref_mask = np.zeros((height, width))
    for winy in range(0, NY, patch_n):
        for winx in range(0, NX, patch_n):
            i = NY - patch_n if winy > NY - patch_n else winy
            j = NX - patch_n if winx > NX - patch_n else winx
            ref_mask[i*ratio:i*ratio+rp, j*ratio+rp-1] = 1
            ref_mask[i*ratio+rp-1, j*ratio:j*ratio+rp] = 1
    ref_mask = np.flip(ref_mask, axis=0)
    # 영상 불러오기
    LR_img = rgb2gray(plt.imread('../0. Datas and Preprocessing/IMAGE/r%d/IMG_r%d_s%d.png' % (ratio, ratio, subject)))
    ref_img = rgb2gray(plt.imread('../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s%d.png' % subject))
    print(np.sum(ref_img))

    inter_img = imresize(LR_img, (height, width), 'bilinear')/255
    Compare('bilinear', ref_img, inter_img)
    print(np.sum(inter_img))

    for comp_path in [
        f"ResNeT_MSE/subject{subject}.png", f"ResNeT_MSE/subject{subject}_Quilt2.png",
        f"ResNeT_MAE/subject{subject}.png", f"ResNeT_MAE/subject{subject}_Quilt2.png",
        f"SRGAN/subject{subject}.png", f"SRGAN/subject{subject}_Quilt2.png",
        # f"SRGAN_BVTV/subject{subject}.png", f"SRGAN_BVTV/subject{subject}_Quilt2.png"
    ]:
        comp_img = rgb2gray(plt.imread("../3. Neural Network/Models/" + comp_path))
        Compare(comp_path, ref_img, comp_img)
        if "Quilt" in comp_path:
            bound_path = comp_path[:-4] + "_boundary.png"
            bound_img = plt.imread("../3. Neural Network/Models/" + bound_path)
            mask = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    if bound_img[i, j, 0] == 1 and bound_img[i,j,1] ==0 and bound_img[i,j,2] ==0:
                        mask[i, j] = 1
        else:
            mask = ref_mask
        print(["PBV(ref) : ", Get_Boundary_Variation(ref_img, mask), "PBV : ", Get_Boundary_Variation(comp_img, mask)])
        print(np.sum(comp_img))

