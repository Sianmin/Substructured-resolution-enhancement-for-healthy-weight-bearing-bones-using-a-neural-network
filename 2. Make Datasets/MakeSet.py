# 최종적으로 Dis 채널 6개, LR patch, HR patch 하나씩 구성되게 만들어야 함.
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py

[ratio, patch_n, step] = [10, 8, 2]
[useDis, useSED, showTest] = [False, True,  False]
Adjpatch_n = 8
subject_n = 11

if __name__  == '__main__':
    [height, width] = [2080, 1883]
    [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
    resolution = 0.05*ratio
    index = 0

    HR_shape = (patch_n*ratio, patch_n*ratio, 1)
    data_n = math.ceil(NY/step)*math.ceil(NX/step)*subject_n
    print("Total Data set {}".format(data_n))

    with h5py.File("Dataset_r%d_p%d_adj_SED.hdf5" % (ratio, ratio*patch_n), 'w') as f:
        f.create_dataset('LR', (data_n, patch_n+2*Adjpatch_n, patch_n+2*Adjpatch_n, 1), dtype='float')
        f.create_dataset('HR', (data_n, patch_n*ratio, patch_n*ratio, 1) , dtype='float')
        LR_set = f['LR']
        HR_set = f['HR']
        if useDis:
            f.create_dataset('Dis', (data_n, patch_n+2*Adjpatch_n+1, patch_n+2*Adjpatch_n+1, 6), dtype='float')
            Dis_set = f['Dis']
        if useSED:
            f.create_dataset('SED', (data_n, patch_n+2*Adjpatch_n, patch_n+2*Adjpatch_n, 3), dtype='float')
            SED_set = f['SED']

        for subject in range(1, subject_n + 1):
            LR_DV = np.zeros((NY, NX, 1))
            HR_DV = np.zeros((height, width, 1))
            if useDis: Displacement = np.zeros((NY+1, NX+1, 6))
            if useSED: SED = np.zeros((NY, NX, 3))

            with open("../0. Datas and Preprocessing/DV/r%d/DV_r%d_s%d.DAT" % (ratio, ratio, subject), 'r') as LR_file:
                for i in range(NY):
                    LR_DV[i, :, 0] = list(map(eval,LR_file.readline().split()))
            with open("../0. Datas and Preprocessing/DV/r1/DV_r1_s%d.DAT" % subject, 'r') as HR_file:
                for i in range(height):
                    HR_DV[i, :, 0] = list(map(eval,HR_file.readline().split()))
            if useDis:
                for load_case in range(1,4):
                    with open("../1. FEA/r%d/s%d/DISPLACEMENT_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r') as dis_file:
                        while 1:
                            lines = dis_file.readline().split()
                            if not lines: break
                            [node_x, node_y, dx, dy] = [int(eval(lines[0])/resolution), int(eval(lines[1])/resolution), eval(lines[2]), eval(lines[3])]
                            if node_x != 0 and node_y != 0:
                                Displacement[node_y, node_x, (load_case-1)*2:load_case*2] =[dx, dy]
            if useSED:
                for load_case in range(1,4):
                    with open("../1. FEA/r%d/s%d/SED_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r') as dis_file:
                        while 1:
                            lines = dis_file.readline().split()
                            if not lines: break
                            [elem_x, elem_y, val] = [eval(lines[0]), eval(lines[1]), eval(lines[2])]
                            elem_x = int((elem_x - resolution/2)/resolution)
                            elem_y = int((elem_y - resolution/2)/resolution)
                            SED[elem_y, elem_x, load_case - 1] = val
            # Windowing

            for winy in range(0, NY, step):
                for winx in range(0, NX, step):
                    i = NY - patch_n if winy > NY - patch_n else winy
                    j = NX - patch_n if winx > NX - patch_n else winx
                    [Lox, Ldx, Pox, Pdx, Loy, Ldy, Poy, Pdy]= [j-Adjpatch_n, j+patch_n+Adjpatch_n, 0, patch_n + 2*Adjpatch_n, i-Adjpatch_n, i+patch_n+Adjpatch_n, 0, patch_n + 2*Adjpatch_n]
                    [HPox, HPdx, HPoy, HPdy, HRox, HRdx, HRoy, HRdy] = [0, patch_n*ratio, 0, patch_n*ratio, j*ratio, (j+patch_n)*ratio, i*ratio, (i+patch_n)*ratio]
                    LR_patch = np.zeros((patch_n + 2*Adjpatch_n, patch_n + 2*Adjpatch_n, 1))
                    HR_patch = np.zeros((patch_n*ratio, patch_n*ratio, 1))
                    if useDis: Dis_patch = np.zeros((patch_n + 2*Adjpatch_n+1, patch_n + 2*Adjpatch_n+1, 6))
                    if useSED: SED_patch = np.zeros((patch_n + 2 * Adjpatch_n, patch_n + 2 * Adjpatch_n, 3))

                    if Lox < 0:
                        [Lox, Ldx, Pox, Pdx] = [0, j+patch_n+Adjpatch_n, patch_n-j, 2*patch_n+Adjpatch_n]
                    elif Ldx > NX - 1:
                        [Lox, Ldx, Pox, Pdx] = [j-Adjpatch_n, NX, 0, NX + Adjpatch_n -j]
                    if Loy < 0:
                        [Loy, Ldy, Poy, Pdy] = [0, i+patch_n+Adjpatch_n, patch_n-i, 2*patch_n+Adjpatch_n]
                    elif Ldy > NY - 1:
                        [Loy, Ldy, Poy, Pdy] = [i-Adjpatch_n, NY, 0, NY+Adjpatch_n-i]

                    if (j+patch_n)*ratio >= width:
                        [HPdx, HRdx] = [width - j * ratio, width]
                    if (i+patch_n)*ratio >= height:
                        [HPdy, HRdy] = [height - i * ratio, height]
                    LR_patch[Poy:Pdy, Pox:Pdx, :] = LR_DV[Loy:Ldy, Lox:Ldx]
                    HR_patch[HPoy:HPdy, HPox:HPdx, :] = HR_DV[HRoy:HRdy, HRox:HRdx]

                    LR_set[index, :, :, :] = LR_patch
                    HR_set[index, :, :, :] = HR_patch
                    if useSED:
                        SED_patch[Poy:Pdy, Pox:Pdx, :] = SED[Loy:Ldy, Lox:Ldx]
                        SED_set[index, :, :, :] = SED_patch
                    if useDis:
                        Dis_patch[Poy:Pdy + 1, Pox:Pdx + 1, :] = Displacement[Loy:Ldy + 1, Lox:Ldx + 1, :]
                        Dis_set[index, :, :, :] = Dis_patch
                    index += 1
                    if index % 1000 == 0: print("Make {}th set".format(index))

                    if showTest:
                        plt.gray()
                        plt.subplot(131)
                        plt.imshow(LR_patch[:, :, 0], vmin=0, vmax=1, origin='lower')
                        plt.subplot(132)
                        plt.imshow(HR_patch[:, :, 0], vmin=0, vmax=1, origin='lower')
                        plt.subplot(133)
                        plt.imshow(SED_patch[:, :, 0], origin='lower')
                        plt.show(block=False)
                        plt.pause(0.5)