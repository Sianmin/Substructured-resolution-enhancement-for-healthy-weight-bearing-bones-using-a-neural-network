from keras.utils.io_utils import HDF5Matrix
import keras
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import h5py

[height, width] = [2080, 1883]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def LoadingDatasets(ratio, patch_n, train_ratio, useDis = False, useSED = False):
    rp = ratio*patch_n
    # Data 불러오기 전처리
    filename = f"../2. Make Datasets/Dataset_r{ratio}_p{rp}.hdf5"
    LR_set = HDF5Matrix(filename, 'LR')[:]
    HR_set = HDF5Matrix(filename, 'HR')[:]

    datanum = int(np.shape(LR_set)[0])
    trainnum = int(datanum * train_ratio)
    testnum = int(datanum * (1-train_ratio))
    print(f"Total: {datanum}\tTrain: {trainnum}\tTest: {testnum}")

    # trainset
    LR_set_train = LR_set[0:trainnum, :, :, :]
    LR_set_test = LR_set[trainnum:(trainnum + testnum), :, :, :]
    HR_set_train = HR_set[0:trainnum, :, :, :]
    HR_set_test= HR_set[trainnum:(trainnum + testnum), :, :, :]

    [BC_train, BC_test, SED_train, SED_test] = [0, 0, 0, 0]

    if useDis:
        Dis_set = HDF5Matrix(filename, 'Dis')[:]
        BC_train = Dis_set[0:trainnum, :, :, :]
        BC_test = Dis_set[trainnum:(trainnum+testnum), :, :, :]
    if useSED:
        SED_set = HDF5Matrix(filename, 'SED')[:]
        SED_train = SED_set[0:trainnum, :, :, :]
        SED_test = SED_set[trainnum:(trainnum+testnum), :, :, :]
    return LR_set_train, LR_set_test, HR_set_train, HR_set_test, BC_train, BC_test ,SED_train, SED_test

def Load_LRDV(ratio, subject):
    # y가 0 이면 밑에서 부터임.
    with open(f"../0. Datas and Preprocessing/DV/r{ratio}/DV_r{ratio}_s{subject}.DAT", 'r') as LR_file:
        [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
        LR_DV = np.zeros((NY, NX))
        for i in range(NY):
            LR_DV[i, :] = list(map(eval, LR_file.readline().split()))
    return LR_DV

def Load_Dis(ratio, subject):
    [NY, NX] = [math.floor(height / ratio), math.floor(width / ratio)]
    Displacement = np.zeros((NY + 1, NX + 1, 6))
    for load_case in range(1,4):
        with open(f"../1. FEA/r{ratio}/s{subject}/Displacement_r{ratio}_s{subject}_c{load_case}", 'r') as dis_file:
            while 1:
                lines = dis_file.readline().split()
                if not lines: break
                [node_x, node_y, dx, dy] = [int(eval(lines[0])), int(eval(lines[1])), eval(lines[2]), eval(lines[3])]
                if node_x != 0 and node_y != 0:
                    Displacement[node_y, node_x, (load_case-1)*2:load_case*2] =[dx, dy]
    return Displacement

def Load_SED(ratio, subject):
    [NY, NX] = [math.floor(height / ratio), math.floor(width / ratio)]
    resolution = 0.05*ratio
    SED = np.zeros((NY, NX, 3))
    for load_case in range(1,4):
        with open(f"../1. FEA/r{ratio}/s{subject}/SED_r{ratio}_s{subject}_c{load_case}", 'r') as dis_file:
            while 1:
                lines = dis_file.readline().split()
                if not lines: break
                [elem_x, elem_y, val] = [eval(lines[0]), eval(lines[1]), eval(lines[2])]
                elem_x = int((elem_x - resolution / 2) / resolution)
                elem_y = int((elem_y - resolution / 2) / resolution)
                SED[elem_y, elem_x, load_case-1] = val
    return SED

class LiveDrawing(keras.callbacks.Callback):
    def __init__(self, filepath, LR_set_train, HR_set_train, LR_set_test, HR_set_test, BC_train, BC_test, SED_train, SED_test, ratio, patch_n, epoch_show=False, useDis=False, useSED=False):
        self.filepath = filepath
        self.LR_set_train, self.HR_set_train = LR_set_train, HR_set_train
        self.LR_set_test, self.HR_set_test = LR_set_test, HR_set_test
        self.useDis, self.useSED = useDis, useSED
        self.epoch_show, self.rp, self.ratio, self.patch_n = epoch_show, ratio * patch_n, ratio, patch_n
        self.NY, self.NX = math.floor(height / self.ratio), math.floor(width / self.ratio)  # 각 축에 대해 나열되어야 하는 패치의 개수
        if useDis: self.BC_train, self.BC_test = BC_train, BC_test
        if useSED: self.SED_train, self.SED_test = SED_train, SED_test

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

    def on_epoch_end_GAN(self, epoch, model, logs={}):
        plt.gray()
        showProg = False
        if showProg:
            a = 1

        num1 = int(random.random() * self.LR_set_train.shape[0])
        num2 = int(random.random() * self.LR_set_test.shape[0])
        num3 = int(random.random() * self.LR_set_test.shape[0])
        if self.useDis:
            result1 = model.predict([np.expand_dims(self.LR_set_train[num1, :, :, :], axis = 0), np.expand_dims(self.BC_train[num1, :, :, :], axis=0)])
            result2 = model.predict([np.expand_dims(self.LR_set_test[num2, :, :, :], axis=0), np.expand_dims(self.BC_test[num2, :, :, :], axis=0)])
            result3 = model.predict([np.expand_dims(self.LR_set_test[num3, :, :, :], axis=0), np.expand_dims(self.BC_test[num3, :, :, :], axis=0)])
        else:
            result1 = model.predict(np.expand_dims(self.LR_set_train[num1,:,:,:], axis = 0))
            result2 = model.predict(np.expand_dims(self.LR_set_test[num2,:,:,:], axis = 0))
            result3 = model.predict(np.expand_dims(self.LR_set_test[num3,:,:,:], axis = 0))

        for i in range(1, 12):
            total_img = self.predictModel(model, i)
            plt.imsave(self.filepath + f'{epoch:02d}-predict_subject{i}.png', total_img[:, :, 0], vmin=0, vmax=1,
                       origin='lower')
        fig = plt.figure(1)
        plt.subplot(1, 4, 4); plt.imshow(total_img[:, :, 0], vmin=0, vmax=1, origin='lower')
        plt.subplot(3, 4, 1) ;plt.imshow(self.LR_set_train[num1, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 2) ;plt.imshow(self.HR_set_train[num1, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 3) ;plt.imshow(result1[0, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 5) ;plt.imshow(self.LR_set_test[num2, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 6) ;plt.imshow(self.HR_set_test[num2, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 7) ;plt.imshow(result2[0, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 9) ;plt.imshow(self.LR_set_test[num3, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 10) ;plt.imshow(self.HR_set_test[num3, :, :, 0], vmin=0, vmax=1)
        plt.subplot(3, 4, 11) ;plt.imshow(result3[0, :, :, 0], vmin=0, vmax=1)
        if self.epoch_show: plt.show(block=False)
        fig.savefig(f"{self.filepath}EPOCH-{epoch:02d}.png", dpi=1200)

    def predictModel(self, model, subject):
        [NY, NX, ratio, patch_n, rp] = [self.NY, self.NX, self.ratio, self.patch_n, self.rp]
        HR_DV = np.zeros((height, width, 1))
        LR_DV = Load_LRDV(ratio, subject)
        LR_patch = np.zeros((1, patch_n, patch_n, 1))
        if self.useDis:
            Dis_patch = np.zeros((1, patch_n+1, patch_n+1, 6))
            Displacement = Load_Dis(ratio, subject)
        if self.useSED:
            SED_patch = np.zeros((1, patch_n, patch_n, 3))
            SED = Load_SED(ratio, subject)

        for winy in range(0, NY, patch_n):
            for winx in range(0, NX, patch_n):
                i = NY - patch_n if winy > NY - patch_n else winy
                j = NX - patch_n if winx > NX - patch_n else winx
                # 주변 부 모두 trabecualr면 해당 patch predict하고 아니면 그냥 넣기
                HR_DV[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio, 0] = LR_DV[i, j]
                LR_patch[0, :, :, 0] = LR_DV[i:i+patch_n, j:j+patch_n]
                if self.useDis:
                    Dis_patch[0, :, :, :] = Displacement[i:i+patch_n+1,j:j+patch_n+1]
                    predict_patch = model.predict([LR_patch, Dis_patch])
                else: predict_patch = model.predict(LR_patch)
                predict_patch = np.reshape(predict_patch, (1, rp, rp))
                HR_DV[i * ratio: i*ratio + rp, j * ratio: j*ratio +rp, 0] = predict_patch
        return HR_DV