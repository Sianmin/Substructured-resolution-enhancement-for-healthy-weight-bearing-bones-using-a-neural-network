import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from MetricsCommon import rgb2gray, readESB, readNSB
import WriteTecplot
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/1. FEA")
from FEA_Bone import FEA_BONE

height, width = 2080, 1883
ry, rx, side_length = 325, 1250, 300
ry = height - ry - side_length

if __name__ == '__main__':
    def printStructuralBehavior(measure, ref, target, load_case):
        ref_elem_n = np.sum(ref > 0)
        target_elem_n = np.sum(target > 0)
        print("%s - Load case:%d" % (measure, load_case))
        print("SSIM\n%f" % ssim(ref, target))
        print("RMSE\n%f" % np.sqrt(mse(ref, target)))
        print("----원본\t\tTarget----")
        print("Maximum\n%f\t\t%f" % (np.max(ref), np.max(target)))
        print("Minimum\n%f\t\t%f" % (np.min(ref), np.min(target)))
        print("Total Sum\n%f\t\t%f" % (np.sum(ref), np.sum(target)))
        print("Average\n%f\t\t%f\n" % (np.sum(ref) / ref_elem_n, np.sum(target) / target_elem_n))
    def MoveFiles(name, subject):
        os.makedirs("FEA/%s" % name, exist_ok=True)
        for load_case in range(1, case_n + 1):
            import shutil
            shutil.move("Displacement%d" % load_case, "FEA/%s/Displacement_s%d_c%d" % (name, subject, load_case))
            shutil.move("VonStress%d" % load_case, "FEA/%s/VonStress_s%d_c%d" % (name, subject, load_case))
            shutil.move("SED%d" % load_case, "FEA/%s/SED_s%d_c%d" % (name, subject, load_case))

    [height, width] = [2080, 1883]
    case_n = 1
    subject = 1

    ref_img = rgb2gray(plt.imread('../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s%d.png' % subject))
    # FEA_BONE(np.flip(ref_img, axis = 0), subject, 1, getVonStress = True, getSED= True)
    # MoveFiles("Ref", subject)
    # WriteTecplot.WriteTecplotFiles_ROI("Ref", subject)
    print("Ref Total BM: %f" % np.sum(ref_img))
    Von_ref_Stress = readESB("Ref", subject, "VonStress")
    Von_ref_SED = readESB("Ref", subject, "SED")
    #

    for TargetModel, Target_path in [("Bilinear", "../4. Metrics/bilinear.png"),
    ("JJNet", "../3. Neural Network/Models/Completed/JJNet-09-04-13-31/predict_subject1.png"),
    ("JJNet-Quilt", "../3. Neural Network/Models/Completed/JJNet-09-04-13-31/CNNQuilt_subject1_DV.png"),
        ("SRGAN_BVTV_AUTOENCODER-10-09-00-05",
         "../3. Neural Network/Models/SRGAN_BVTV_AUTOENCODER-10-09-00-05/05-predict_subject1.png"),
        ("SRGAN_BVTV_AUTOENCODER-10-09-00-05_Quilt",
         "../3. Neural Network/Models/SRGAN_BVTV_AUTOENCODER-10-09-00-05/Quilt_subject1_DV.png")]:
        target_img = rgb2gray(plt.imread(Target_path))

        print(TargetModel)
        print("Target Total BM: %f" % np.sum(target_img))

        # '''FEA 돌리기'''
        FEA_BONE(np.flip(target_img, axis = 0), 1, getVonStress = True, getSED= True)
        MoveFiles(TargetModel, subject)
        WriteTecplot.WriteTecplotFiles_ROI(TargetModel, subject)
        WriteTecplot.WriteTecplotFiles(TargetModel, subject)


        Von_target = readESB(TargetModel, subject, "VonStress")
        for load_case in range(1, case_n+1):
            printStructuralBehavior("VonMisesStress", Von_ref_Stress[ry:ry+side_length, rx:rx+side_length, load_case-1], Von_target[ry:ry+side_length, rx:rx+side_length, load_case - 1], load_case)
            printStructuralBehavior("VonMisesStress Total", Von_ref_Stress[:, :, load_case - 1], Von_target[:,:, load_case - 1], load_case)

        Von_target = readESB(TargetModel, subject, "SED")
        for load_case in range(1, case_n+1):
            printStructuralBehavior("Strain Energy Density", Von_ref_SED[ry:ry+side_length, rx:rx+side_length, load_case - 1], Von_target[ry:ry+side_length, rx:rx+side_length, load_case - 1], load_case)
            printStructuralBehavior("Strain Energy Density Total", Von_ref_SED[:,:, load_case - 1], Von_target[:,:, load_case - 1], load_case)

