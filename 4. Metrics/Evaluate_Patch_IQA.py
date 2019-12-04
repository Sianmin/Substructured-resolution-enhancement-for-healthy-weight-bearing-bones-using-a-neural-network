import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from scipy.misc import imresize

[height, width] = [2080, 1883]
[ratio, patch_n] = [10, 8]
[NX, NY] = [math.floor(width / ratio), math.floor(height / ratio)]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == '__main__':
    plt.gray()
    for mode in ["training", "validation"]:
        ssim_list, psnr_list, bv_err_list = [], [], []
        if mode == "training": subjects = range(1, 8)
        else: subjects = range(8, 12)
        for subject in subjects:
            LR_img = rgb2gray(plt.imread(f'../0. Datas and Preprocessing/IMAGE/r{ratio}/IMG_r{ratio}_s{subject}.png'))
            ref = rgb2gray(plt.imread(f'../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s{subject}.png'))

            img_path =f"../3. Neural Network/Models/SRGAN/subject{subject}.png"
            if img_path != "bilinear": target = rgb2gray(plt.imread(img_path))
            else: target= imresize(LR_img, (height, width), 'bilinear')/255

            for winy in range(0, NY, patch_n):
                for winx in range(0, NX, patch_n):
                    i = NY - patch_n if winy > NY - patch_n else winy
                    j = NX - patch_n if winx > NX - patch_n else winx
                    ref_patch = ref[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
                    target_patch = target[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
                    if np.mean(ref_patch) > 0.10 and np.mean(ref_patch)< 0.90: # 꽉 차 있거나 비어있는 패치를 제외하고 BV/TV가 0.1이상 0.9 이하만 비교
                        ssim_list.append(ssim(ref_patch, target_patch))
                        psnr_list.append(psnr(ref_patch, target_patch, data_range=1))
                        bv_err_list.append(np.abs(np.sum(ref_patch)-np.sum(target_patch)))

        ssim_list = np.asarray(ssim_list)
        psnr_list = np.asarray(psnr_list)
        bv_err_list = np.asarray(bv_err_list)

        print(img_path)
        print("%f\t%f\t%f\t%f\t%f"%(max(psnr_list), min(psnr_list), np.mean(psnr_list), np.median(psnr_list), np.std(psnr_list)))
        print("%f\t%f\t%f\t%f\t%f"%(max(ssim_list), min(ssim_list), np.mean(ssim_list), np.median(ssim_list), np.std(ssim_list)))
        print("%f\t%f\t%f\t%f\t%f"%(max(bv_err_list), min(bv_err_list), np.mean(bv_err_list), np.median(bv_err_list), np.std(bv_err_list)))

