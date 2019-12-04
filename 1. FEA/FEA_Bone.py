# /CWD, D:\Project\Trabecualr_Subsutrcutred_Deep_learning
'''
MPA system
Space: mm
Force: N
modulus: MPa
'''
import numpy as np
from datetime import datetime
import math, time, os

def unit_vec(vector):
    return vector / (vector**2).sum()**0.5

def AssignLoad(filename, forces, NX, NY, FH_NODE, GT_NODE, FHC, FHS, GTC, GT_ref_node1, A_Flag):
    [FORCE, THETA] = [forces[0], forces[1]]
    [FX, FY] = [FORCE*math.sin(-np.radians(THETA)), -FORCE*math.cos(np.radians(THETA))]
    force_node = []
    R_VEC = unit_vec(FHC - FHS)
    C_VEC = [-math.sin(np.radians(THETA)), -math.cos(np.radians(THETA))]  # 합력 방향
    RANGE = math.acos(np.dot(R_VEC, C_VEC))

    with open(filename, 'w') as ansys_LOAD:
        ansys_LOAD.write("NSEL, ALL\nFDELE, ALLL\n") # 이전에 부여한 하중 지우기.
        # FH_NODE 하중 부여
        for i in range(len(FH_NODE)):
            node_now = FH_NODE[i, :]
            U_VEC = unit_vec(FHC - node_now)
            U_THETA = math.acos(np.dot(U_VEC, C_VEC))
            if abs(U_THETA) <= abs(RANGE):
                FN = FORCE * math.cos(np.radians(90 / RANGE * U_THETA))
                force_node.append([node_now[0], node_now[1], FN*U_VEC[0], FN*U_VEC[1]])
        force_node = np.asarray(force_node)

        # rescaling
        FXSUM = np.sum(force_node[:, 2])
        FYSUM = np.sum(force_node[:, 3])
        force_node[:, 2] = force_node[:, 2] * FX / FXSUM
        force_node[:, 3] = force_node[:, 3] * FY / FYSUM

        # Write and plotting
        for i in range(len(force_node)):
            noden = force_node[i, 0] + force_node[i, 1] * (NX + 1) + 1
            ansys_LOAD.write("F, %d, FX, %f\n" % (noden, force_node[i, 2]))
            ansys_LOAD.write("F, %d, FY, %f\n" % (noden, force_node[i, 3]))

        # Greater Trochanter
        [FORCE, THETA] = [forces[2], forces[3]]
        [FX, FY] = [FORCE * math.sin(np.radians(THETA)), FORCE * math.cos(np.radians(THETA))]
        R_VEC = unit_vec(GTC - GT_ref_node1)
        C_VEC = [math.sin(np.radians(THETA)), math.cos(np.radians(THETA))]  # 합력 방향
        force_node = []
        for i in range(len(GT_NODE)):
            node_now = GT_NODE[i, :]
            U_VEC = unit_vec(GTC - node_now)
            FN = FORCE * np.dot(U_VEC, np.array([0, 1]))
            force_node.append([node_now[0], node_now[1], FN*C_VEC[0], FN*C_VEC[1]])
        force_node = np.asarray(force_node)

        # rescaling
        FXSUM = np.sum(force_node[:, 2])
        FYSUM = np.sum(force_node[:, 3])
        force_node[:, 2] = force_node[:, 2] * FX / FXSUM
        force_node[:, 3] = force_node[:, 3] * FY / FYSUM

        # Writing and plotting
        for i in range(len(force_node)):
            noden = force_node[i, 0] + force_node[i, 1] * (NX + 1) + 1
            ansys_LOAD.write("F, %d, FX, %f\n" % (noden, force_node[i, 2]))
            ansys_LOAD.write("F, %d, FY, %f\n" % (noden, force_node[i, 3]))

        # Boundary Condition
        ansys_LOAD.write("NSEL, S, LOC, Y, 0\nD, ALL, UX, 0\nD, ALL, UY, 0\n")

def getPoints(A_Flag, ratio):
    # Extract outer nodes
    [NY, NX] = A_Flag.shape
    A_NODE = np.zeros((NY + 1, NX + 1))
    PAD_Flag = np.pad(A_Flag, 1, 'constant', constant_values=0)
    for i in range(NY):
        for j in range(NX):
            if A_Flag[i, j] == 1:
                for ik in range(2):
                    for jk in range(2):
                        A_NODE[i + ik, j + jk] = 0
                        for ikk in range(2):
                            for jkk in range(2):
                                if PAD_Flag[i + ik + ikk, j + jk + jkk] == 0:
                                    A_NODE[i + ik, j + jk] += 1
    Outer_NODE = np.greater_equal(A_NODE, 2)  # order: y,x
    Outer_NODE_NE = []
    for i in range(NY + 1):
        for j in range(NX + 1):
            if Outer_NODE[i, j]:
                Outer_NODE_NE.append([i, j])  # order: y, x
    Outer_NODE_NE = np.asarray(Outer_NODE_NE)

    FHC = np.asarray([round(1429 / ratio), round(1633 / ratio)])  # Femoral head center, order: x y
    GTC = np.asarray([round(463 / ratio), round(1640 / ratio)])  # Greater Trochanter Center, order: x, y
    FHS = np.asarray([round(1103 / ratio), round(1931 / ratio)])  # Femoral head load의 distribution 시작점.

    # Extract FH nodes
    for i in range(FHC[0], 0, -1):  # x 탐색
        if Outer_NODE[FHC[1], i]:
            FH_ref_node1 = [i, FHC[1]]  # FHC의 왼쪽 첫 번째 점, order: x, y
            break
    for i in range(FHC[1], 0, -1):  # y 탐색
        if Outer_NODE[i, FHC[0]]:
            FH_ref_node2 = [FHC[0], i]  # FHC의 아래쪽 첫 번째 점, order: x, y
            break
    FH_NODE = []
    for i in range(NY + 1):
        for j in range(NX + 1):
            if Outer_NODE[i, j]:
                if (j >= FH_ref_node1[0] and i >= FHC[1]) or (j >= FHC[0] and i >= FH_ref_node2[1]):
                    FH_NODE.append([j, i])  # x, y 순
    FH_NODE = np.asarray(FH_NODE)

    # Extract GT nodes
    for i in range(GTC[0], 0, -1):  # x 탐색
        if Outer_NODE[GTC[1], i]:
            GT_ref_node1 = [i, GTC[1]]  # GTC의 왼쪽 첫 번째 점, order: x, y
            break
    for i in range(GTC[0], NY):  # 탐색
        if Outer_NODE[GTC[1], i]:
            GT_ref_node2 = [i, GTC[1]]  # GTC의 오른쪽 첫 번째 점, order: x, y
            break
    GT_NODE = []
    for i in range(NY + 1):
        for j in range(NX + 1):
            if Outer_NODE[i, j]:
                if (j >= GT_ref_node1[0] and i >= GTC[1] and j <= GT_ref_node2[0]):
                    GT_NODE.append([j, i])  # x, y 순
    GT_NODE = np.asarray(GT_NODE)

    return FH_NODE, GT_NODE, FHC, FHS, GTC, GT_ref_node1

def FEA_BONE(A_BMD, ratio, getDis = True, getVonStress = False, getSED = False, saveScript=False):
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
    [CORTICAL_E, TRABECULAR_E, TH, E0] = [22500, 15000, 0.1, 15000]
    [W, H] = [9.415, 10.400]
    [NY, NX] = A_BMD.shape
    resolution = 0.05*ratio
    #Flag 정보 불러오기
    with open("../0. Datas and Preprocessing/FLAG/FLAG_r%d.DAT" % ratio, 'r') as Bone_Flag_fid:
        A_Flag = np.zeros((NY, NX))
        for i in range(NY):
            A_Flag[i, :] = list(map(int, Bone_Flag_fid.readline().split()))
    # ANSYS_input 파일 만들기
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
                          "/SOLVE\n*USE, ANSYS_LOAD1\n*USE, ANSYS_POST1\n")
                          # "/SOLVE\n*USE, ANSYS_LOAD2\n*USE, ANSYS_POST2\n"
                          # "/SOLVE\n*USE, ANSYS_LOAD3\n*USE, ANSYS_POST3\n")

    # MAT 생성
    with open("ANSYS_MAT", 'w') as ansys_MAT:
        for i in range(NY):
            for j in range(NX):
                ele = i * NX + j + 1
                ansys_MAT.write(f"MP, EX, {ele}, {E0*(A_BMD[i, j]**3) + 0.001*E0} \nMP, PRXY, {ele}, 0.3\n") # for topology optimization
                # if ratio > 0:  # Continuum material property
                #     if A_BMD[i, j] > 0.84:
                #         ele = i * NX + j + 1
                #         ansys_MAT.write(
                #             "MP, EX, %d, %f\nMP, PRXY, %d, 0.3\n" % (ele, E0 * 0.1908 * (2 * A_BMD[i, j]) ** 2.39, ele))
                #     elif A_BMD[i, j] >= 0:
                #         ele = i * NX + j + 1
                #         ansys_MAT.write(
                #             "MP, EX, %d, %f\nMP, PRXY, %d, 0.3\n" % (ele, E0 * 0.3044 * (2 * A_BMD[i, j]) ** 1.49 + 0.1, ele))
                # else:
                #     if A_Flag[i, j] == 1:
                #         ele = i * NX + j + 1
                #         ansys_MAT.write(
                #             "MP, EX, %d, %f\nMP, PRXY, %d, 0.3\n" % (ele, CORTICAL_E, ele))
                #     elif A_Flag[i, j] == 2:
                #         ele = i * NX + j + 1
                #         if A_BMD[i, j] >= TH:
                #             ansys_MAT.write(
                #                 "MP, EX, %d, %f\nMP, PRXY, %d, 0.3\n" % (ele, TRABECULAR_E * (A_BMD[i, j] ** 3), ele))
                #         else:
                #             ansys_MAT.write(
                #                 "MP, EX, %d, %f\nMP, PRXY, %d, 0.3\n" % (ele, TRABECULAR_E * A_BMD[i, j] / 100+10, ele))

    # Node 생성
    with open("ANSYS_NODE", 'w') as ansys_NODE:
        N_NODE = np.zeros((NY + 1, NX + 1))
        for i in range(NY):
            for j in range(NX):
                if A_Flag[i, j]:
                    N_NODE[i:i + 2, j:j + 2] = [[1, 1], [1, 1]]
        for i in range(NY + 1):
            for j in range(NX + 1):
                if N_NODE[i, j]:
                    ansys_NODE.write("N, %d, %f, %f\n" % (i * (NX + 1) + j + 1, j*resolution, i*resolution))

    # MESH 생성
    with open("ANSYS_MESH", 'w') as ansys_MESH:
        for i in range(NY):
            for j in range(NX):
                ele = i * NX + j + 1
                n1 = (NX + 1) * i + j + 1
                [n2, n3, n4] = [n1 + 1, n1 + NX + 2, n1 + NX + 1]
                if A_Flag[i, j]:
                    ansys_MESH.write("MAT, %d\n" % ele)
                    ansys_MESH.write("E, %d, %d, %d, %d\n" % (n1, n2, n3, n4))

    # 하중 지정을 위한 특정 점들 구하기
    [FH_NODE, GT_NODE, FHC, FHS, GTC, GT_ref_node1] = getPoints(A_Flag, ratio)

    # 하중 조건 부여
    AssignLoad("ANSYS_LOAD1", [2317, 24, 703, 28], NX, NY, FH_NODE, GT_NODE, FHC, FHS, GTC, GT_ref_node1, A_Flag)
    # AssignLoad("ANSYS_LOAD2", [1158, -15, 351, -8], NX, NY, FH_NODE, GT_NODE, FHC, FHS, GTC, GT_ref_node1, A_Flag)
    # AssignLoad("ANSYS_LOAD3", [1548, 56, 468, 35], NX, NY, FH_NODE, GT_NODE, FHC, FHS, GTC, GT_ref_node1, A_Flag)

    # Post-processing 생성
    for case in range(1, 2):
        with open("ANSYS_Post%d" % case, 'w') as ansys_POST:
            ansys_POST.write("ALLSEL\n"
                             "EQSLV, PCG, 1E-6\n"
                             "SOLVE\n"
                             "FINISH\n"
                             "/POST1\n"
                             "*VGET,NODE_X,NODE,ALL,LOC,X, , ,2\n"
                             "*VGET,NODE_Y,NODE,ALL,LOC,Y, , ,2\n"
                             "*VGET,ELEM_X,ELEM,ALL,CENT,X\n"
                             "*VGET,ELEM_Y,ELEM,ALL,CENT,Y\n"
                             "*GET, NE, ELEMENT, 0, COUNT,\n"
                             "*VGET, EARRAY, ELEM, , ELIST,\n")
            if getDis:
                ansys_POST.write("*VGET,DIS_X,NODE,ALL,U,X, , ,2\n"
                                 "*VGET,DIS_Y,NODE,ALL,U,Y, , ,2\n"
                                 "/OUTPUT, Displacement%d\n"
                                 "*VWRITE, NODE_X(1), NODE_Y(1), DIS_X(1), DIS_Y(1)\n"
                                 "(F8.3, F8.3, F15.7, F15.7)\n"
                                 "/OUTPUT\n" % case)
            if getVonStress:
                ansys_POST.write("*GET,NE,ELEM,0,COUNT\n"
                                 "*DIM, VONSTRESS, ARRAY, NE\n"
                                 "ETABLE, VONSTRESS, S, EQV\n"
                                 "*DO, i, 1, NE\n"
                                 "*GET, VONSTRESS(i), ETAB, 1, ELEM, EARRAY(i)\n"
                                 "*ENDDO\n"
                                 "/OUTPUT, VonStress%d\n"
                                 "*VWRITE, ELEM_X(1), ELEM_Y(1), VONSTRESS(1)\n"
                                 "(F8.3, F8.3, F15.7)\n"
                                 "/OUTPUT\n" % case)
            if getSED:
                ansys_POST.write("*GET,NE,ELEM,0,COUNT\n"
                                 "*DIM, SED, ARRAY, NE\n"
                                 "ETABLE, SED, SEDN\n"
                                 "*DO, i, 1, NE\n"
                                 "*GET, SED(i), ETAB, 2, ELEM, EARRAY(i)\n"
                                 "*ENDDO\n"
                                 "/OUTPUT, SED%d\n"
                                 "*VWRITE, ELEM_X(1), ELEM_Y(1), SED(1)\n"
                                 "(F8.3, F8.3, F15.7)\n"
                                 "/OUTPUT\n" % case)
    # # Run FEA
    file_name = "ANSYS_INPUT"
    TIME_START = datetime.now()
    print("START FEA")
    os.system("ansys140 -b -np 2 -i %s.ans -o %s_output.ans" % (file_name, file_name))
    TIME_END = datetime.now()
    TIME = TIME_END - TIME_START
    print(["FEA TIME", TIME])
    ClearFiles(saveScript)

if __name__ == '__main__':
    # Parameters
    for ratio in [10]:
        [NX, NY] = [math.floor(1883/ratio), math.floor(2080/ratio)]
        os.makedirs("r%d" % (ratio), exist_ok=True)
        for subject in range(1, 12):
            print("Subject {}".format(subject))
            # Subject의 DV 불러오기
            with open("../0. Datas and Preprocessing/DV/r%d/DV_r%d_s%d.DAT" % (ratio, ratio, subject), 'r') as Bone_BMD_fid:
                A_BMD = np.zeros((NY, NX))
                for i in range(NY):
                    A_BMD[i, :] = Bone_BMD_fid.readline().split()
            FEA_BONE(A_BMD, ratio, getVonStress = True, getSED= True)

            # Displacement들 옮기기
            os.makedirs("r%d" % (ratio), exist_ok=True)
            os.makedirs("r%d/s%d" % (ratio, subject), exist_ok=True)
            import shutil
            for load_case in range(1, 2):
                shutil.move("Displacement%d"%load_case, "r%d/s%d/Displacement_r%d_s%d_c%d"%(ratio, subject, ratio, subject, load_case))
                shutil.move("SED%d" % load_case,"r%d/s%d/SED_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case))

