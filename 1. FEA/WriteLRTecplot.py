import numpy as np
import math
import os, sys

[height, width] = [2080, 1883]
case_n = 3
ratio = 10

def Load_Flag(ratio):
    with open("../0. Datas and Preprocessing/FLAG/FLAG_r%d.DAT" % ratio, 'r') as Flag_file:
        [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
        Flag = np.zeros((NY, NX))
        for i in range(NY):
            Flag[i, :] = list(map(eval, Flag_file.readline().split()))
    return Flag

def WriteTecplotFiles(ratio, subject, onlyTra = False):
    [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
    Dis_ref = np.zeros((NY+1, NX+1, 2, case_n))
    Dis_Flag = np.zeros((NY+1, NX+1))
    resolution = 0.05*ratio
    for load_case in range(1, case_n + 1):
        with open("r%d/s%d/Displacement_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r') as ref_fid:
            while 1:
                lines = ref_fid.readline().split()
                if not lines: break
                [elem_x, elem_y, dx, dy] = [math.floor(eval(lines[0])/resolution), math.floor(eval(lines[1])/resolution), eval(lines[2]), eval(lines[3])]
                Dis_ref[elem_y, elem_x, :, load_case-1] = [dx, dy]
                Dis_Flag[elem_y, elem_x] = 1

    step = int(NX/3)

    for load_case in range(1, case_n+1):
        with open("r%d/s%d/Displacement_r%d_s%d_c%d_Plot.PLT" % (ratio, subject, ratio, subject, load_case), 'w') as fid:
            fid.write("TITLE = \"Displacement -- ratio: %d -- subject: %d -- Load Case: %d\"\n"%(ratio, subject, load_case))
            fid.write("VARIABLES = \"X\", \"Y\", \"Dx\", \"Dy\"\n")
            fid.write("ZONE T=\"FRAME\"\n")
            for i in range(0, NY+1, step):
                for j in range(0, NX+1, step):
                    if Dis_Flag[i, j]:
                        fid.write("%f, %f, %f, %f\n" % (j*resolution, i*resolution, Dis_ref[i, j, 0, load_case-1], Dis_ref[i, j, 1, load_case-1]))

    '''Strain Energy Density'''
    SED_ref = np.zeros((NY, NX, case_n))
    Flag = np.zeros((NY, NX))
    for load_case in range(1, case_n+1):
        with open("r%d/s%d/SED_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r') as ref_fid:
            while 1:
                lines = ref_fid.readline().split()
                if not lines: break
                [elem_x, elem_y, val] = [math.floor(eval(lines[0])/resolution), math.floor(eval(lines[1])/resolution), eval(lines[2])]
                SED_ref[elem_y, elem_x, load_case - 1] = val
                Flag[elem_y, elem_x] = 1
    elem_n = sum(sum(Flag))

    for load_case in range(1, 4):
        with open("r%d/s%d/SED_r%d_s%d_c%d_PLOT.PLT" % (ratio, subject, ratio, subject, load_case), 'w') as fid:
            fid.write("TITLE = \"SED -- ratio: %d -- subject: %d -- Load Case: %d\"\n"%(ratio, subject, load_case))
            fid.write("VARIABLES = \"X\", \"Y\", \"SED\"\n")
            fid.write("ZONE N=%d, E=%d, F=FEPOINT, ET=QUADRILATERAL\n" % (elem_n*4, elem_n))
            for i in range(NY):
                for j in range(NX):
                    if Flag[i, j]:
                        for ii in range(2):
                            for jj in range(2):
                                fid.write("%f, %f, %f\n" % ((j+jj)*resolution, (i+ii)*resolution, SED_ref[i, j, load_case-1]))
            for i in range(height*width):
                fid.write("%d %d %d %d\n" % (i*4+1,i*4+2,i*4+3,i*4+4))

if __name__ == '__main__':
    for subject in range(1, 12):
        WriteTecplotFiles(ratio, subject)