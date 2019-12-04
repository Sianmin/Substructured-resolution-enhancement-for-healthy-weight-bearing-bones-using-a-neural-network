import matplotlib.pyplot as plt
import numpy as np
import os
import math
from MetricsCommon import rgb2gray, readESB, readNSB

side_length = 300
resolution = 0.05
noden = (side_length + 1) ** 2
ROI_name = ['Femoral Head']
ROI_origin = [(1250, 325)]

def FEA_ROI(BMD, saveScript = False):
    def ClearFiles(saveScript):
        file_list = os.listdir()
        file_list_conf = [file for file in file_list if file.startswith("file")]
        for file in file_list_conf:
            os.remove(file)
        if not saveScript:
            file_list_conf = [file for file in file_list if file.startswith("ANSYS")]
            for file in file_list_conf:
                os.remove(file)
    ClearFiles(saveScript)
    E0 = 15000
    [NY, NX] = BMD.shape

    with open("ANSYS_INPUT.ans", 'w') as ansys_input:
        ansys_input.write("/GRAPH, POWER\n"
                          "/UNITS, MPA\n"
                          "/PREP7\n"
                          "/NOPR\n"
                          "ET, 1, PLANE182\n"
                          "KEYOPT, 1, 1, 2\n"
                          "KEYOPT, 1, 3, 3\n"
                          "*USE, ANSYS_MAT\n"
                          "*USE, ANSYS_NODE\n"
                          "*USE, ANSYS_MESH\n\nTYPE, 1\nREAL, 1\nR, 1,1\n\n"
                          "NSEL, ALL\n\n"
                          "/SOLVE\n*USE, ANSYS_LOAD1\n*USE, ANSYS_POST1\n"
                          "/SOLVE\n*USE, ANSYS_LOAD2\n*USE, ANSYS_POST2\n"
                          "/SOLVE\n*USE, ANSYS_LOAD3\n*USE, ANSYS_POST3\n")
    with open("ANSYS_MAT", 'w') as ansys_MAT:
        for i in range(NY):
            for j in range(NX):
                ele = i * NX + j + 1
                ansys_MAT.write(f"MP, EX, {ele}, {E0 * (BMD[i, j] ** 3) + 0.001 * E0} \nMP, PRXY, {ele}, 0.3\n")  # for topology optimization
    with open("ANSYS_NODE", 'w') as ansys_NODE:
        for i in range(NY + 1):
            for j in range(NX + 1):
                ansys_NODE.write("N, %d, %f, %f\n" % (i * (NX + 1) + j + 1, j*resolution, i*resolution))
    with open("ANSYS_MESH", 'w') as ansys_MESH:
        for i in range(NY):
            for j in range(NX):
                ele = i * NX + j + 1
                n1 = (NX + 1) * i + j + 1
                [n2, n3, n4] = [n1 + 1, n1 + NX + 2, n1 + NX + 1]
                ansys_MESH.write("MAT, %d\n" % ele)
                ansys_MESH.write("E, %d, %d, %d, %d\n" % (n1, n2, n3, n4))
    with open("ANSYS_LOAD1", 'w') as ansys_LOAD:
        ansys_LOAD.write("NSEL, ALL\nDDELE, ALL\n")
        ansys_LOAD.write("NSEL, S, LOC, Y, 0\nD, ALL, UY, 0\n")
        ansys_LOAD.write("NSEL, S, LOC, X, 0\nD, ALL, UX, 0\n")
        ansys_LOAD.write("NSEL, S, LOC, Y, {}\nD, ALL, UY, 0\n".format((NY+1)*resolution))
        ansys_LOAD.write("NSEL, S, LOC, X, {}\nD, ALL, UX, {}\n".format((NX+1)*resolution, side_length*resolution))
    with open("ANSYS_LOAD2", 'w') as ansys_LOAD:
        ansys_LOAD.write("NSEL, ALL\nDDELE, ALL\n")
        ansys_LOAD.write("NSEL, S, LOC, Y, 0\nD, ALL, UY, 0\n")
        ansys_LOAD.write("NSEL, S, LOC, X, 0\nD, ALL, UX, 0\n")
        ansys_LOAD.write("NSEL, S, LOC, Y, {}\nD, ALL, UY, {}\n".format((NY+1)*resolution, side_length*resolution))
        ansys_LOAD.write("NSEL, S, LOC, X, {}\nD, ALL, UX, 0\n".format((NX+1)*resolution))
    with open("ANSYS_LOAD3", 'w') as ansys_LOAD:
        ansys_LOAD.write("NSEL, ALL\nDDELE, ALL\n")
        ansys_LOAD.write("NSEL, S, LOC, Y, 0\nD, ALL, UX, 0\n")
        ansys_LOAD.write("NSEL, S, LOC, X, 0\nD, ALL, UY, 0\n")
        ansys_LOAD.write("NSEL, S, LOC, Y, {}\nD, ALL, UX, {}\n".format((NY+1)*resolution, side_length*resolution))
        ansys_LOAD.write("NSEL, S, LOC, X, {}\nD, ALL, UY, 0\n".format((NX+1)*resolution))
    with open("ANSYS_POST1", 'w') as ansys_POST:
        ansys_POST.write("ALLSEL\n"
                         "EQSLV, PCG, 1E-6\n"
                         "SOLVE\n"
                         "FINISH\n"
                         "/POST1\n"
                         "*DIM, RF, array, {}\n"
                         "*DO, I, 1, {}\n"
                         "*GET, RF(I), NODE, 1+ {}*{}, RF, FX\n"
                         "*ENDDO\n"
                         "/OUTPUT, Reaction1\n"
                         "*VWRITE, RF(1)\n"
                         "(F15.6)\n"
                         "/OUTPUT\n"
                         "FINISH".format(side_length+1, side_length+1, side_length+1,"(I-1)"))
    with open("ANSYS_POST2", 'w') as ansys_POST:
        ansys_POST.write("ALLSEL\n"
                         "EQSLV, PCG, 1E-6\n"
                         "SOLVE\n"
                         "FINISH\n"
                         "/POST1\n"
                         "*DIM, RF, array, {}\n"
                         "*DO, I, 1, {}\n"
                         "*GET, RF(I), NODE, I, RF, FY\n"
                         "*ENDDO\n"
                         "/OUTPUT, Reaction2\n"
                         "*VWRITE, RF(1)\n"
                         "(F15.6)\n"
                         "/OUTPUT\n"
                         "FINISH".format(side_length+1, side_length+1))
    with open("ANSYS_POST3", 'w') as ansys_POST:
        ansys_POST.write("ALLSEL\n"
                         "EQSLV, PCG, 1E-6\n"
                         "SOLVE\n"
                         "FINISH\n"
                         "/POST1\n"
                         "*DIM, RF, array, {}\n"
                         "*DO, I, 1, {}\n"
                         "*GET, RF(I), NODE, I, RF, FX\n"
                         "*ENDDO\n"
                         "/OUTPUT, Reaction3\n"
                         "*VWRITE, RF(1)\n"
                         "(F15.6)\n"
                         "/OUTPUT\n"
                         "FINISH".format(side_length+1, side_length+1))

    # Run FEA
    file_name = "ANSYS_INPUT"
    os.system("ansys140 -b -np 2 -i ANSYS_INPUT.ans -o ANSYS_output.ans")
    ClearFiles(saveScript)
def getApparentElasticModulus():
    E = np.zeros((3))
    for case in range(1, 4):
        with open("Reaction{}".format(case), 'r') as reaction:
            forces = np.zeros((side_length+1, 1))
            for i in range(side_length+1):
                forces[i] = eval(reaction.readline())
            force_sum = abs(np.sum(forces))
            if case == 3:
                force_sum = force_sum * 4/math.pi # 전단 응력의 경우 side length 만큼 옮겨주면 전단 변형률이 pi/4
            E[case-1] = force_sum/(side_length*resolution)
    print("{}\t{}\t{}".format(E[0], E[1], E[2]))
def getMorphometricIndices(ROI):
    # BVTV
    print("{}%".format(np.sum(ROI)/(side_length**2)*100))

if __name__ == '__main__':
    plt.gray()
    for subject in range(1, 2):
        print(f"Subject {subject}")
        total_img = plt.imread('../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s{}.png'.format(subject))
        ref_img = rgb2gray(total_img)
        for roi_index in range(len(ROI_name)):
            print("Region {}".format(ROI_name[roi_index]))
            [ox, oy] = ROI_origin[roi_index]
            ROI_ref = ref_img[oy:oy + side_length, ox:ox + side_length]
            FEA_ROI(np.flip(ROI_ref, axis=0))
            getApparentElasticModulus()
            getMorphometricIndices(ROI_ref)
            for TargetModel, TargetPath in [("Bilinear", f"../4. Metrics/bilinear{subject}.png"),
                                            ("ResNet-MSE", f"../3. Neural Network/Models/ResNet_MSE/subject{subject}.png"),
                                            ("ResNet-MSE-Quilt", f"../3. Neural Network/Models/ResNet_MSE/subject{subject}_Quilt2.png"),
                                            ("ResNet-MAE", f"../3. Neural Network/Models/ResNet_MAE/subject{subject}.png"),
                                            ("ResNet-MAE-Quilt", f"../3. Neural Network/Models/ResNet_MAE/subject{subject}_Quilt2.png"),
                                            ("SRGAN", f"../3. Neural Network/Models/SRGAN/subject{subject}.png"),
                                            ("SRGAN-Quilt", f"../3. Neural Network/Models/SRGAN/subject{subject}_Quilt2.png")]:
                    target_img = rgb2gray(plt.imread(TargetPath))
                    ROI_target = target_img[oy:oy+side_length, ox:ox+side_length]
                    FEA_ROI(np.flip(ROI_target, axis=0), saveScript = True)
                    print(TargetModel)
                    getApparentElasticModulus()
                    getMorphometricIndices(ROI_target)

                    # where_img = plt.imread('../0. Datas and Preprocessing/IMAGE/r1/IMG_r1_s{}.png'.format(subject))
                    # where_img[oy:oy+side_length, ox:ox+side_length, 1:3] = 0
                    # total_img[oy:oy+side_length, ox:ox+side_length, 1:3] = 0
                    # plt.imsave('Total_{}_Reference.png'.format(ROI_name[roi_index]), where_img)
                    # plt.imsave('Total_Reference.png', total_img)
                    plt.imsave('{}_ref.png'.format(ROI_name[roi_index]), ROI_ref)
                    plt.imsave('{}_{}.png'.format(ROI_name[roi_index], TargetModel), ROI_target)
