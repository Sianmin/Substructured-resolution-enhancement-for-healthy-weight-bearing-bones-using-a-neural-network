from keras.utils.io_utils import HDF5Matrix
import keras
import matplotlib.pyplot as plt
import numpy as np
import math

[height, width] = [2080, 1883]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def LoadingDatasets(ratio, patch_n, useDis = False, useSED = False):
    rp = ratio*patch_n
    # Data 불러오기 전처리
    filename = f"../2. Make Datasets/Dataset_r{ratio}_p{rp}.hdf5"
    LR_set = HDF5Matrix(filename, 'LR')[:]
    datanum = LR_set.shape[0]
    print(f"Training Data: {datanum}")

    HR_set = HDF5Matrix(filename, 'HR')[:]
    Dis_set, SED_set = [], []
    if useDis: Dis_set = HDF5Matrix(filename, 'Dis')[:]
    if useSED: SED_set = HDF5Matrix(filename, 'SED')[:]
    return LR_set,  HR_set, Dis_set, SED_set

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
    def __init__(self, filepath, LR_set_train, HR_set_train, BC_train,SED_train, ratio, patch_n, epoch_show=False, useDis=False, useSED=False):
        self.filepath = filepath
        self.LR_set_train, self.HR_set_train = LR_set_train, HR_set_train
        self.useDis, self.useSED = useDis, useSED
        self.epoch_show, self.rp, self.ratio, self.patch_n = epoch_show, ratio * patch_n, ratio, patch_n
        self.NY, self.NX = math.floor(height / self.ratio), math.floor(width / self.ratio)  # 각 축에 대해 나열되어야 하는 패치의 개수
        if useDis: self.BC_train = BC_train
        if useSED: self.SED_train = SED_train

    def on_train_begin(self, logs={}):
        self.losses, self.val_losses, self.accuracy, self.val_accuracy = [], [], [], []

    def on_epoch_end(self, epoch, logs={}):
        plt.gray()
        for i in range(1, 12):
            total_img = self.predictModel(self.model, i)
            plt.imsave(self.filepath + f'{epoch:02d}-predict_subject{i}.png', total_img[:, :, 0], vmin=0, vmax=1, origin='lower')

    def on_epoch_end_GAN(self, epoch, model):
        plt.gray()
        for i in range(1, 12):
            total_img = self.predictModel(model, i)
            plt.imsave(self.filepath + f'{epoch:02d}-predict_subject{i}.png', total_img[:, :, 0], vmin=0, vmax=1, origin='lower')

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
                ender = 0
                i = NY - patch_n if winy > NY - patch_n else winy
                j = NX - patch_n if winx > NX - patch_n else winx

                HR_DV[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio, 0] = LR_DV[i, j]
                LR_patch[0, :, :, 0] = LR_DV[i:i+patch_n, j:j+patch_n]

                if self.useDis:
                    Dis_patch[0, :, :, :] = Displacement[i:i+patch_n+1,j:j+patch_n+1]
                    predict_patch = model.predict([LR_patch, Dis_patch])
                else: predict_patch = model.predict(LR_patch)

                predict_patch = np.reshape(predict_patch, (1, rp, rp))
                HR_DV[i * ratio: i*ratio + rp, j * ratio: j*ratio +rp, 0] = predict_patch
        return HR_DV