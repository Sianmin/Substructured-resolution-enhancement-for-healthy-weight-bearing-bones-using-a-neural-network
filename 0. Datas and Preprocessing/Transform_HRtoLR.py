import numpy as np
import matplotlib.pyplot as plt
import math

[height, width] = [2080, 1883]
'''해당 Ratio의 FLAG 만들기'''
for ratio in [8, 10]:
    [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
    # 저해상도 Flag 만들기 모두 Tra면 Tra, 하나라도 cortical 있으면 cor, 그게 아니면 0 (없음)
    with open("Bone_FLAG.DAT", 'r') as Bone_Flag_fid:
        A_Flag = np.zeros((height, width))
        for i in range(height):
            A_Flag[height-1-i, :] = list(map(int, Bone_Flag_fid.readline().split()))

    # LR Flag 만들기
    LR_Flag = np.zeros((NY, NX))
    for i in range(NY):
        for j in range(NX):
            temp = A_Flag[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
            if np.sum(np.equal(temp, np.ones((ratio,ratio))*2)) == ratio**2:
                LR_Flag[i, j] = 2
            elif np.sum(np.equal(temp, np.ones((ratio,ratio))*1)) >= 1:
                LR_Flag[i, j] = 1
            else:
                LR_Flag[i, j] = 0

    with open(f"FLAG/FLAG_r{ratio}.DAT", 'w') as Flag_file:
        for i in range(NY):
            for j in range(NX):
                Flag_file.write("%d "%LR_Flag[i, j])
            Flag_file.write("\n")

'''해당 ratio의 Subject별 DV 만들기'''
for ratio in [1, 8, 10]:
    [NY, NX] = [math.floor(height / ratio), math.floor(width / ratio)]
    '''해당 ratio의 Subject별 DV 만들기'''
    for TEST in range (1, 12):
        # HR 불러오기
        with open(f"../A. 대퇴골자료\CASE {TEST}\CASE{TEST}_XDEN3_HR.DAT", 'r') as file:
            Total = np.zeros((height, width))
            while 1:
                temp = (file.readline().split())
                if not temp: break
                [temp1, temp2, temp3] = [eval(temp[0]), eval(temp[1]), eval(temp[2])]
                Total[temp2 - 1, temp1 - 1] = temp[2]

        # 저해상화 LR_DV 계산
        LR_DV = np.zeros((NY, NX))
        for i in range(NY):
            for j in range(NX):
                temp = Total[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio]
                LR_DV[i, j] = np.sum(temp)/ (ratio**2)

        # 저해상화 LR_DV 저장하기
        with open(f"DV/r{ratio}/DV_r{ratio}_s{TEST}.DAT", 'w') as file:
            for i in range(NY):
                for j in range(NX):
                    file.write(f"{LR_DV[i, j]} ")
                file.write("\n")

        # 영상으로도 저장
        with open(f"DV/r{ratio}/DV_r{ratio}_s{TEST}.DAT", 'r') as file:
            Total = np.zeros((math.floor(height/ratio), math.floor(width/ratio)))
            for i in range(math.floor(height/ratio)):
                temp = (file.readline().split())
                Total[i, :] = temp
        plt.gray()
        plt.imsave(f"IMAGE/r{ratio}/IMG_r{ratio}_s{TEST}.png", Total, origin='lower')