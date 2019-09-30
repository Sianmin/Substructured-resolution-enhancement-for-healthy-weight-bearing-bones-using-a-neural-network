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
    filename = '../2. Make Datasets/Dataset_r%d_p%d_t1.hdf5' % (ratio, rp)
    LR_set = HDF5Matrix(filename, 'LR')[:]
    HR_set = HDF5Matrix(filename, 'HR')[:]

    datanum = int(np.shape(LR_set)[0])
    trainnum = int(datanum * train_ratio)
    testnum = int(datanum * (1-train_ratio))
    print("Total: %d\tTrain: %d\tTest: %d"%(datanum, trainnum, testnum))

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
    with open("../0. Datas and Preprocessing/DV/r%d/DV_r%d_s%d.DAT" % (ratio, ratio, subject), 'r') as LR_file:
        [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]
        LR_DV = np.zeros((NY, NX))
        for i in range(NY):
            LR_DV[i, :] = list(map(eval, LR_file.readline().split()))
    return LR_DV

def Load_Dis(ratio, subject):
    [NY, NX] = [math.floor(height / ratio), math.floor(width / ratio)]
    Displacement = np.zeros((NY + 1, NX + 1, 6))
    for load_case in range(1,4):
        with open("../1. FEA/r%d/s%d/Displacement_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r') as dis_file:
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
        with open("../1. FEA/r%d/s%d/SED_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r') as dis_file:
            while 1:
                lines = dis_file.readline().split()
                if not lines: break
                [elem_x, elem_y, val] = [eval(lines[0]), eval(lines[1]), eval(lines[2])]
                elem_x = int((elem_x - resolution / 2) / resolution)
                elem_y = int((elem_y - resolution / 2) / resolution)
                SED[elem_y, elem_x, load_case-1] = val
    return SED

class LiveDrawing(keras.callbacks.Callback):
    def __init__(self, filepath, LR_set_train, HR_set_train, LR_set_test, HR_set_test, BC_train, BC_test, SED_train, SED_test, ratio, patch_n, epoch_show=False, isGAN=False, useDis=False, useSED=False):
        self.filepath = filepath
        self.LR_set_train, self.HR_set_train = LR_set_train, HR_set_train
        self.LR_set_test, self.HR_set_test = LR_set_test, HR_set_test
        self.useDis = useDis
        self.useSED = useSED
        if useDis:
            self.BC_train = BC_train
            self.BC_test = BC_test
        if useSED:
            self.SED_train = SED_train
            self.SED_test = SED_test
        self.epoch_show = epoch_show
        self.rp = ratio*patch_n
        self.ratio, self.patch_n = [ratio, patch_n]
        self.NY, self.NX = [math.floor(height/self.ratio), math.floor(width/self.ratio)] # 각 축에 대해 나열되어야 하는 패치의 개수
        self.mode = 1 # 1: Generator, 2: Quilter

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        from IPython.display import clear_output
        print("End of Epoch")

        epochs = range(1, (epoch + 1) + 1)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        plt.gray()

        '''Loss 보여주기'''
        showProg = False
        if showProg:
            clear_output(wait=True)
            plt.subplot(2, 1, 1)
            plt.title('Training loss and accuracy')
            plt.plot(epochs, self.losses, 'b', label='Training loss')
            plt.plot(epochs, self.val_losses, 'r', label='Validation loss')
            plt.ylabel('loss')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.ylabel('acc')
            plt.plot(epochs, self.accuracy, 'b', label='Training acc')
            plt.plot(epochs, self.val_accuracy, 'r', label='Validation acc')
            plt.xlabel('Epochs')
            plt.legend()
            plt.show()

        ''' 전체 모델 Predict'''
        for i in range(1, 12):
            total_img = self.predictModel(self.model, i)
            plt.imsave(self.filepath + '%02d-predict_subject%d.png'%(epoch+1, i), total_img[:, :, 0], vmin=0, vmax=1, origin='lower')
        if self.epoch_show:
            num1 = int(random.random() * self.LR_set_train.shape[0])
            num2 = int(random.random() * self.LR_set_test.shape[0])
            num3 = int(random.random() * self.LR_set_test.shape[0])
            if self.useDis:
                result1 = self.model.predict([np.expand_dims(self.LR_set_train[num1, :, :, :], axis=0),
                                         np.expand_dims(self.BC_train[num1, :, :, :], axis=0)])
                result2 = self.model.predict([np.expand_dims(self.LR_set_test[num2, :, :, :], axis=0),
                                         np.expand_dims(self.BC_test[num2, :, :, :], axis=0)])
                result3 = self.model.predict([np.expand_dims(self.LR_set_test[num3, :, :, :], axis=0),
                                         np.expand_dims(self.BC_test[num3, :, :, :], axis=0)])
            elif self.useSED:
                result1 = self.model.predict([np.expand_dims(self.LR_set_train[num1, :, :, :], axis=0),np.expand_dims(self.SED_train[num1, :, :, :], axis=0)])
                result2 = self.model.predict([np.expand_dims(self.LR_set_train[num2, :, :, :], axis=0),np.expand_dims(self.SED_train[num2, :, :, :], axis=0)])
                result3 = self.model.predict([np.expand_dims(self.LR_set_train[num3, :, :, :], axis=0),np.expand_dims(self.SED_train[num3, :, :, :], axis=0)])
            fig = plt.figure(1)
            plt.subplot(1, 4, 4); plt.imshow(total_img[:, :, 0], vmin=0, vmax=1, origin='lower')
            plt.subplot(3, 4, 1); plt.imshow(self.LR_set_train[num1, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 2); plt.imshow(self.HR_set_train[num1, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 3); plt.imshow(result1[0, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 5); plt.imshow(self.LR_set_test[num2, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 6); plt.imshow(self.HR_set_test[num2, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 7); plt.imshow(result2[0, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 9); plt.imshow(self.LR_set_test[num3, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 10); plt.imshow(self.HR_set_test[num3, :, :, 0], vmin=0, vmax=1)
            plt.subplot(3, 4, 11); plt.imshow(result3[0, :, :, 0], vmin=0, vmax=1)
            plt.show(block=False)
            fig.savefig(self.filepath + "EPOCH-%2d.png" % (epoch), dpi=1200)
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
            plt.imsave(self.filepath + '%02d-predict_subject%d.png' % (epoch, i), total_img[:, :, 0], vmin=0, vmax=1,
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
        fig.savefig(self.filepath + "EPOCH-%2d.png"%(epoch),dpi=1200)

    # def create_QN_Data(self, GN, filepath, subjects):
    #     [patch_xn, patch_yn] = [math.floor(width / self.rp), math.floor(height / self.rp)]
    #     subject_crop_n = 2*patch_xn*patch_yn-patch_xn-patch_yn
    #     total_n = subject_crop_n * len(subjects)
    #     print([subject_crop_n, total_n  ,"%sDataset_QN.hdf5"%(filepath)])
    #     with h5py.File("%sDataset_QN.hdf5" % (filepath), 'w') as f:
    #         f.create_dataset('In', (total_n, self.rp, self.rp, 1), dtype='float')
    #         f.create_dataset('Out', (total_n, self.rp, self.rp, 1), dtype='float')
    #         In_set = f['In']
    #         Out_set = f['Out']
    #         index = 0
    #         for subject in subjects:
    #             HR_DV = Load_LRDV(1, subject)
    #             pred_DV = self.predictModel(GN, subject)
    #
    #             HR_crop = self.crop_for_QN(HR_DV)
    #             pred_crop = self.crop_for_QN(pred_DV)
    #             In_set[index*subject_crop_n:(index+1)*subject_crop_n, :, :, :] = pred_crop[:, :, :, :]
    #             Out_set[index*subject_crop_n:(index+1)*subject_crop_n: ,: ,:, :] = HR_crop[:, :, :, :]
    #             index += 1
    #     In_set_o =  HDF5Matrix("%sDataset_QN.hdf5" % (filepath), 'In')[:]
    #     Out_set_o =  HDF5Matrix("%sDataset_QN.hdf5" % (filepath), 'Out')[:]
    #     return In_set_o, Out_set_o
    #
    # def crop_for_QN(self, DV):
    #     rp = self.rp
    #     [patch_xn, patch_yn] = [math.floor(width/rp), math.floor(height/rp)]
    #     Cropset = np.zeros((2*patch_xn*patch_yn-patch_xn-patch_yn, rp, rp, 1))
    #     patch = np.zeros((rp, rp, 1))
    #     index = 0
    #     for i in range(patch_yn):
    #         for j in range(patch_xn - 1):
    #             if (j+1)*rp + int(rp/2) <= width:
    #                 patch = DV[i*rp:(i+1)*rp, j*rp+int(rp/2):(j + 1) * rp + int(rp/2)]
    #                 if patch.ndim <= 2: patch = np.expand_dims(patch, axis = 2)
    #                 Cropset[index, :, :, :] = patch
    #                 index += 1
    #     for i in range(patch_yn - 1):
    #         for j in range(patch_xn):
    #             if (i+1)*rp+int(rp/2) <= height:
    #                 patch = DV[i * rp + int(rp/2):(i + 1) * rp + int(rp/2), j * rp :(j + 1) * rp]
    #                 patch = np.rot90(patch)
    #                 if patch.ndim <= 2: patch = np.expand_dims(patch, axis = 2)
    #                 Cropset[index, :, :, :] = patch
    #                 index += 1
    #     return Cropset
    #
    # def attach_for_QN(self, DV, crops):
    #     index = 0
    #     [rp, patch_n] = [self.rp, self.patch_n]
    #     [patch_xn, patch_yn] = [math.floor(width / self.rp), math.floor(height / rp)]
    #     for i in range(patch_yn):
    #         for j in range(patch_xn - 1):
    #             if (j + 1) * rp + int(rp / 2) <= width:
    #                 DV[i*rp:(i+1)*rp, j*rp+int(rp/2):(j+1)*rp+int(rp/2), :] = crops[index, :, :, :]
    #                 index += 1
    #     for i in range(patch_yn - 1):
    #         for j in range(patch_xn):
    #             if (i + 1) * rp + int(rp / 2) <= height:
    #                 crop = crops[index, :, :, :]
    #                 crop = np.rot90(crop, 3)
    #                 DV[i*rp+int(rp/2):(i+1)*rp+int(rp/2), j*rp:(j+1)*rp, :] = crop
    #                 index += 1
    #     return DV
    #
    # def predcit_Quilt(self, GN, QN):
    #     [rp, patch_n] = [self.rp, self.patch_n]
    #     [patch_xn, patch_yn] = [math.floor(width / self.rp), math.floor(height / rp)]
    #     plt.gray()
    #     for subject in range(1, 12):
    #         predict_HR = self.predictModel(GN, subject)
    #         for i in range(patch_yn):
    #             for j in range(patch_xn - 1):
    #                 if (j + 1) * rp + int(rp / 2) <= width:
    #                     predict_patch  = QN.predict(np.expand_dims(predict_HR[i*rp:(i+1)*rp, j*rp+int(rp/2):(j+1)*rp+int(rp/2), :], axis=0))
    #                     predict_patch = np.squeeze(predict_patch, axis=0)
    #                     # predict_HR[i*rp:(i+1)*rp, j*rp+int(rp/2)+int(rp/4):(j+1)*rp+int(rp/2)-int(rp/4), :] = predict_patch[:, int(rp/4):int(3*rp/4):, :]
    #                     predict_HR[i*rp:(i+1)*rp, j*rp+int(rp/2):(j+1)*rp+int(rp/2), :] = predict_patch
    #         for i in range(patch_yn - 1):
    #             for j in range(patch_xn):
    #                 if (i + 1) * rp + int(rp / 2) <= height:
    #                     patch = np.rot90(predict_HR[i*rp+int(rp/2):(i+1)*rp+int(rp/2), j*rp:(j+1)*rp, :])
    #                     patch = np.expand_dims(patch, axis = 0)
    #                     patch = np.squeeze(QN.predict(patch), axis=0)
    #                     predict_patch = np.rot90(patch, 3)
    #                     # predict_HR[i * rp + int(rp / 2) + int(rp/4):(i + 1) * rp + int(rp / 2)-int(rp/4), j * rp:(j + 1) * rp, :] = predict_patch[int(rp/4):int(3*rp/4), :, :] # 가운데 부분만 넣기
    #                     predict_HR[i*rp+int(rp/2):(i+1)*rp+int(rp/2), j*rp:(j+1)*rp, :] = predict_patch
    #         plt.imsave(self.filepath + '%02d-quilt_subject%d.png' % (1, subject), predict_HR[:, :, 0], vmin=0, vmax=1, origin='lower')
    #     return predict_HR

    def predictModel(self, model, subject):
        [NY, NX, ratio, patch_n, rp] = [self.NY, self.NX, self.ratio, self.patch_n, self.rp]
        [patch_nx, patch_ny] = [math.floor(NX/patch_n), math.floor(NY/patch_n)]
        HR_DV = np.zeros((height, width, 1))
        LR_DV = Load_LRDV(ratio, subject)
        if self.useDis: Displacement = Load_Dis(ratio, subject)
        if self.useSED:
            SED_patch = np.zeros((3 * patch_n, 3 * patch_n, 3))
            SED = Load_SED(ratio, subject)

        # for winy in range(0, NY, patch_n):
        #     for winx in range(0, NX, patch_n):
        #         i = NY - patch_n if winy > NY - patch_n else winy
        #         j = NX - patch_n if winx > NX - patch_n else winx
        #         # 주변 부 모두 trabecualr면 해당 patch predict하고 아니면 그냥 넣기
        #         HR_DV[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio, 0] = LR_DV[i, j]
        #         LR_patch[0, :, :, 0] = LR_DV[i:i+patch_n, j:j+patch_n]
        #         if self.useDis:
        #             Dis_patch[0, :, :, :] = Displacement[i:i+patch_n+1,j:j+patch_n+1]
        #             predict_patch = model.predict([LR_patch, Dis_patch])
        #         else: predict_patch = model.predict(LR_patch)
        #         predict_patch = np.reshape(predict_patch, (1, rp, rp))
        #         HR_DV[i * ratio: i*ratio + rp, j * ratio: j*ratio +rp, 0] = predict_patch

        '''Adj부분을 위한 Predict Model'''
        for winy in range(0, NY, patch_n):
            for winx in range(0, NX, patch_n):
                i = NY - patch_n if winy > NY - patch_n else winy
                j = NX - patch_n if winx > NX - patch_n else winx
                [Lox, Ldx, Pox, Pdx, Loy, Ldy, Poy, Pdy]= [j-patch_n, j+2*patch_n, 0, 3*patch_n, i-patch_n, i+2*patch_n, 0, 3*patch_n]
                [HPox, HPdx, HPoy, HPdy, HRox, HRdx, HRoy, HRdy] = [0, patch_n*ratio, 0, patch_n*ratio, j*ratio, (j+patch_n)*ratio, i*ratio, (i+patch_n)*ratio]
                LR_patch = np.zeros((3*patch_n, 3*patch_n, 1))

                if Lox < 0:
                    [Lox, Ldx, Pox, Pdx] = [0, j+2*patch_n, patch_n-j, 3*patch_n]
                elif Ldx > NX - 1:
                    [Lox, Ldx, Pox, Pdx] = [j-patch_n, NX, 0, NX + patch_n -j]
                if Loy < 0:
                    [Loy, Ldy, Poy, Pdy] = [0, i+2*patch_n, patch_n-i, 3*patch_n]
                elif Ldy > NY - 1:
                    [Loy, Ldy, Poy, Pdy] = [i-patch_n, NY, 0, NY+patch_n-i]
                if (j+patch_n)*ratio >= width:
                    [HPdx, HRdx] = [width - j * ratio, width]
                if (i+patch_n)*ratio >= height:
                    [HPdy, HRdy] = [height - i * ratio, height]
                HR_DV[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio, 0] = LR_DV[i, j]
                LR_patch[Poy:Pdy, Pox:Pdx, :] = np.expand_dims(LR_DV[Loy:Ldy, Lox:Ldx], axis = 2)
                if self.useSED:
                    SED_patch[Poy:Pdy, Pox:Pdx, :] = SED[Loy:Ldy, Lox:Ldx,:]
                # HR_patch[HPoy:HPdy, HPox:HPdx, :] = HR_DV[HRoy:HRdy, HRox:HRdx]
                predict_patch = model.predict([np.expand_dims(LR_patch,axis = 0), np.expand_dims(SED_patch, axis=0)])
                HR_DV[i * ratio: i*ratio + rp, j * ratio: j*ratio +rp, :] = predict_patch
        return HR_DV

    def EfrosFreeman(self, model, subject, step = 0):
        HR = np.zeros((height, width, 1))
        LR_patch = np.zeros((1, self.patch_n, self.patch_n, 1))
        LR_DV = Load_LRDV(self.ratio, subject)
        if step == 0:
            step = int(self.patch_n/4)

        for i in range(0, self.NY, step):
            for j in range(0, self.NX, step):
                LR_patch[0, :, :, 0] = LR_DV[0:self.patch_n, 0:self.patch_n, 0]
                HR_patch = model.predict(LR_patch)
                if i > 0 and j > 0: # L-shape
                    LR_patch[0, :, :, 0] = LR_DV[i:i+self.patch_n, j:j+self.patch_n, 0]
                    HR_patch = model.predict(LR_patch)
                elif i > 0: # Horizontal
                    continue
                elif j > 0: # Vertical
                    continue
                else: # Initial
                    HR[0:self.rp, 0:self.rp, 0] = HR_patch
