import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from MetricsCommon import rgb2gray, readESB, readNSB
import WriteTecplot
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/1. FEA")
from FEA_Bone import FEA_BONE

if __name__ == '__main__':
    def printStructuralBehavior(measure, ref, target, load_case):
        ref_elem_n = np.sum(ref > 0)
        target_elem_n = np.sum(target > 0)
        print("%s - Load case:%d" % (measure, load_case))
        print("SSIM\n%f" % ssim(ref, target))
        print("MSE\n%f" % mse(ref, target))
        print("----원본\t\tTarget----")
        print("Maximum\n%f\t\t%f" % (np.max(ref), np.max(target)))
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
    case_n = 3
    subject = 1
    epoch = 5

    ref_img = rgb2gray(plt.imread('../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s%d.png' % subject))
    # FEA_BONE(np.flip(ref_img, axis = 0), subject, 1, getVonStress = True, getSED= True)
    # MoveFiles("Ref", subject)
    # WriteTecplot.WriteTecplotFiles("Ref", subject)
    #
    for TargetModel, Target_path in [("SRGAN-09-11-12-47","../3. Neural Network/Models/Completed/%s/%02d-predict_subject%d.png" % ("SRGAN-09-11-12-47", epoch, subject)),
                                     ("CNNquilt","../3. Neural Network/Models/Completed/%s/CNN_Feasibility%d.png" % ("CNNquilt", subject)),
                                     ("JJNet-09-04-13-31", "../3. Neural Network/Models/Completed/%s/%2d-predict_subject%d.png" % ("JJNet-09-04-13-31", epoch, subject)),
                                     ("SRGANquilt","../3. Neural Network/Models/Completed/%s/Feasibility%d.png" % ("SRGANquilt", subject) )]:

        ''' 정보 불러 오기 '''
        target_img = rgb2gray(plt.imread(Target_path))
        print("Ref Total BM: %f" % np.sum(ref_img))
        print("Target Total BM: %f" % np.sum(target_img))

        # '''FEA 돌리기'''
        FEA_BONE(np.flip(target_img, axis = 0), 1, getVonStress = True, getSED= True)
        MoveFiles(TargetModel, subject)

        Von_ref = readESB("Ref", subject, "VonStress")
        Von_target = readESB(TargetModel, subject, "VonStress")
        for load_case in range(1, case_n+1):
            printStructuralBehavior("VonMisesStress", Von_ref[:, :, load_case-1], Von_target[:, :, load_case - 1], load_case)

        Von_ref = readESB("Ref", subject, "SED")
        Von_target = readESB(TargetModel, subject, "SED")
        for load_case in range(1, case_n+1):
            printStructuralBehavior("Strain Energy Density", Von_ref[:, :, load_case - 1], Von_target[:, :, load_case - 1], load_case)

        WriteTecplot.WriteTecplotFiles(TargetModel, subject)