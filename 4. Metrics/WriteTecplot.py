import numpy as np
import math
import os
from MetricsCommon import readESB, readNSB

[height, width] = [2080, 1883]
case_n = 3

def Load_Flag(ratio):
    with open("../0. Datas and Preprocessing/FLAG/FLAG_r%d.DAT" % ratio, 'r') as Flag_file:
        [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
        Flag = np.zeros((NY, NX))
        for i in range(NY):
            Flag[i, :] = list(map(eval, Flag_file.readline().split()))
    return Flag

def WriteTecplotFiles(model, subject, ratio=1, SED=True, VonStress=True, Displacement=False, onlyTra = False):
    resolution = 0.05 * ratio
    [NY, NX]=[math.floor(height/ratio), math.floor(width/ratio)]
    os.makedirs("FEA/%s/Plot" % model, exist_ok=True)

    if onlyTra: Flag = Load_Flag(1)

    '''Von Mises Stress'''
    if VonStress:
        Von_ref = readESB(model, subject, "VonStress")
        Von_Flag = Von_ref > 0
        Von_Flag = Von_Flag[:, :, 0]
        elem_n = np.sum(Von_Flag)

        for load_case in range(1, case_n+1):
            if onlyTra: fn = "FEA/%s/Plot/Tra_VonStress_s%d_c%d_PLOT.PLT" % (model, subject, load_case)
            else: fn = "FEA/%s/Plot/VonStress_s%d_c%d_PLOT.PLT" % (model, subject, load_case)
            with open(fn, 'w') as fid:
                fid.write("TITLE = \"%s -- VonMises -- Load Case: %d\"\n"%(model, load_case))
                fid.write("VARIABLES = \"X (mm)\", \"Y (mm)\", \"Von Mises Stress (MPa)\"\n")
                fid.write("ZONE N=%d, E=%d, F=FEPOINT, ET=QUADRILATERAL\n" % (elem_n*4, elem_n))
                for i in range(NY):
                    for j in range(NX):
                        if Von_Flag[i, j]:
                            for ii in range(2):
                                for jj in range(2):
                                    fid.write("%f, %f, %f\n" % ((j+jj)*resolution, (i+ii)*resolution, Von_ref[i, j, load_case-1]))
                for i in range(elem_n):
                    fid.write("%d %d %d %d\n" % (i*4+1,i*4+2,i*4+3,i*4+4))

    '''Strain Energy Density'''
    if SED:
        SED_ref = readESB(model, subject, "SED")

        for load_case in range(1, 4):
            if onlyTra: fn = "FEA/%s/Plot/Tra_SED_s%d_c%d_PLOT.PLT" % (model, subject, load_case)
            else: fn = "FEA/%s/Plot/SED_s%d_c%d_PLOT.PLT" % (model, subject, load_case)
            with open(fn, 'w') as fid:
                fid.write("TITLE = \"%s -- Strain Energy Density -- Load Case: %d\"\n"%(model, load_case))
                fid.write("VARIABLES = \"X (mm)\", \"Y (mm)\", \"Strain Energy Density (?)\"\n")
                fid.write("ZONE N=%d, E=%d, F=FEPOINT, ET=QUADRILATERAL\n" % (elem_n*4, elem_n))
                for i in range(NY):
                    for j in range(NX):
                        if Von_Flag[i, j]:
                            for ii in range(2):
                                for jj in range(2):
                                    fid.write("%f, %f, %f\n" % ((j+jj)*resolution, (i+ii)*resolution, SED_ref[i, j, load_case-1]))
                for i in range(elem_n):
                    fid.write("%d %d %d %d\n" % (i*4+1,i*4+2,i*4+3,i*4+4))

if __name__ == '__main__':
    WriteTecplotFiles("Ref", 1)
    WriteTecplotFiles("Feasibility", 1)