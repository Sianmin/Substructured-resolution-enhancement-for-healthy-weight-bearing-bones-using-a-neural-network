import tensorflow as tf
from keras import backend as K
from Common import LiveDrawing, Load_LRDV
from Networks import Networks as Networkclass
import matplotlib.pyplot as plt
import numpy as np
import math, heapq
from datetime import datetime

# GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

# Initialization
height, width = 2080, 1883
isGAN, epoch_show = False, False
[ratio, patch_n] = [10, 8]
[epochs, batch_size] = [10, 32]
OV_width = 2

OV_step, OV_rp = patch_n - OV_width, OV_width * ratio
NY, NX = math.floor(height/ratio), math.floor(width/ratio)
rp = ratio * patch_n
plt.gray()

filepath = "Models/SRGAN_BVTV-10-01-14-02/"
gpath = filepath + "20-G.hdf5"
[BC_train, BC_test] = [0, 0]
Networks = Networkclass(ratio, patch_n, batch_size, isGAN)
GN = Networks.Generator_SRGAN_1()
GN.load_weights(gpath)

def cut_boundary_dijkstra(Emap, ori, dest):
    shape = Emap.shape
    cutMap = np.zeros(shape)
    d_min = math.inf
    [mx, my] = [0, 0]
    for o in ori:
        [x, y] = o
        Q = [] # Queue
        Dmap = np.ones(shape) * math.inf # Memorization 위한 map
        Dmap[y, x] = Emap[y, x]
        prevMap = np.zeros(shape) # 가장 빠르게 갈 수 있다고 계산된 이웃 노드 1: 위 2: 오 3: 아래 4: 왼
        heapq.heappush(Q, (Dmap[y, x], x, y))
        while Q:
            current_dist, x, y  = heapq.heappop(Q)
            adj = []
            # 인접 노드 Iteration
            if x > 0:
                if Emap[y, x-1] != math.inf:
                    adj.append((x-1, y, 2))
            if x < shape[1]-1:
                if Emap[y, x + 1] != math.inf:
                    adj.append((x+1, y, 4))
            if y > 0:
                if Emap[y-1, x] != math.inf:
                    adj.append((x, y-1, 1))
            if y < shape[0]-1:
                if Emap[y+1, x] != math.inf:
                    adj.append((x, y+1, 3))
            for (j, i, c) in adj:
                next_dist = current_dist + Emap[i ,j]
                if next_dist < Dmap[i, j]:
                    Dmap[i, j] = next_dist
                    prevMap[i, j] = c
                    heapq.heappush(Q, (Dmap[i,j], j, i))

        # D 중에서 지금 경로 비용보다 제일 작은거 구해냄.
        for d in dest:
            [x, y] = d
            if Dmap[y, x] < d_min:
                [mx, my, d_min] = [x, y, Dmap[y, x]]
                prevMapSave = prevMap # 경로 저장
    # 경로 복구
    while True:
        dir = prevMapSave[my, mx]
        cutMap[my, mx] = 1
        if dir == 0: break
        elif dir == 1: my += 1
        elif dir == 2: mx += 1
        elif dir == 3: my -=1
        elif dir == 4: mx -=1
    return cutMap
def fill_mask(E_mask, Q):
    while Q:
        [x, y] = Q.pop()
        if x > 0:
            if E_mask[y, x - 1] != 1:
                E_mask[y, x - 1] = 1
                Q.append((x - 1, y))
        if x < rp - 1:
            if E_mask[y, x + 1] != 1:
                E_mask[y, x + 1] = 1
                Q.append((x + 1, y))
        if y > 0:
            if E_mask[y - 1, x] != 1:
                E_mask[y - 1, x] = 1
                Q.append((x, y - 1))
        if y < rp - 1:
            if E_mask[y + 1, x] != 1:
                E_mask[y + 1, x] = 1
                Q.append((x, y + 1))
    return E_mask

if __name__ == "__main__":
    # OD initialization
    H_O = []
    H_D = []
    V_O = []
    V_D = []
    L_O = []
    L_D = []
    for i in range(1, OV_rp-1):
        H_O.append((0, i))
        H_D.append((rp - 1, i))
    for i in range(1, OV_rp -1):
        V_O.append((i, 0))
        V_D.append((i, rp - 1))
    for i in range(1, OV_rp - 1):
        L_O.append((i, rp - 1))
        L_D.append((rp - 1, i))

    for subject in range(1, 12):
        LR_DV = np.expand_dims(np.expand_dims(Load_LRDV(ratio, subject), axis = 2), axis = 0)
        HR_DV = np.zeros((height, width))
        cutting = np.zeros((height, width))

        # Windowing 해 나간다.
        for winy in range(0, NY, OV_step):
            for winx in range(0, NX, OV_step):
                i = NY - patch_n if winy > NY - patch_n else winy
                j = NX - patch_n if winx > NX - patch_n else winx
                print((subject, j, i))
                LR_patch = LR_DV[:, i:i+patch_n, j:j+patch_n, :]
                HR_patch = np.squeeze(np.squeeze(GN.predict(LR_patch), axis = 3), axis = 0)
                [GX, GY] = [j*ratio, i*ratio]
                '''L-shape'''
                if i > 0 and j > 0:
                    OV_map = np.zeros((rp, rp))
                    OV_old = np.zeros((rp, rp))
                    OV_new = np.zeros((rp, rp))
                    OV_old = HR_DV[GY:GY+rp, GX:GX+rp]
                    OV_new = HR_patch
                    OV_map = (OV_old - OV_new)**2
                    OV_map[OV_rp:, OV_rp:] = math.inf

                    # boundary 구한 후 boundary 왼쪽 아래 부분은 1로 마스킹
                    start = datetime.now()
                    E_mask = cut_boundary_dijkstra(OV_map, L_O, L_D)
                    print(datetime.now() - start)

                    E_mask_old = E_mask
                    for seed in range(rp):
                        if E_mask[seed, 0] == 0:
                            Q = [(0, seed)]
                            break
                    if not Q:
                        for seed in range(rp):
                            if E_mask[0, seed] == 0:
                                Q=[(seed, 0)]
                                break
                    E_mask = fill_mask(E_mask, Q)

                    # HRDV에 붙여넣기
                    OV_cut = E_mask*OV_old + (1-E_mask)*OV_new
                    HR_DV[GY:GY+rp, GX:GX+OV_rp] = OV_cut[:, :OV_rp]
                    HR_DV[GY:GY+OV_rp, GX:GX+rp] = OV_cut[:OV_rp, :]
                    HR_DV[GY+OV_rp:GY+rp, GX+OV_rp:GX + rp] = HR_patch[OV_rp:, OV_rp:]

                elif i > 0: # Horizontal overlap j == 0
                    OV_old = HR_DV[GY:GY+OV_rp, :rp]
                    OV_new = HR_patch[:OV_rp, :]
                    OV_map = (OV_old - OV_new)**2
                    # boundary 구한 후에 boundary 왼쪽 부분은 1로 마스킹
                    E_mask = cut_boundary_dijkstra(OV_map, H_O, H_D)
                    E_mask_old = E_mask
                    for seed in range(rp):
                        if E_mask[seed, 0] == 0:
                            Q = [(0, seed)]
                            break
                    if not Q:
                        for seed in range(rp):
                            if E_mask[0, seed] == 0:
                                Q=[(seed, 0)]
                                break
                    E_mask = fill_mask(E_mask, Q)
                    # Mask 대로 Overlap 부분 절단 mask의 1인 부분은 old, 0인 부분은 new
                    OV_cut = E_mask*OV_old + (1-E_mask)*OV_new
                    # HRDV에 붙여넣기
                    HR_DV[GY:GY+OV_rp, GX:GX+rp] = OV_cut
                    HR_DV[GY+OV_rp: GY+rp, GX:GX+rp] = HR_patch[OV_rp:, :]

                elif j > 0: # Vertical overlap
                    OV_old = HR_DV[:rp, GX:GX+OV_rp]
                    OV_new = HR_patch[:, 0:OV_rp]
                    OV_map = (OV_old - OV_new)**2
                    # boundary 구한 후에 boundary 아래 부분은 1로 마스킹

                    E_mask = cut_boundary_dijkstra(OV_map, V_O, V_D)

                    E_mask_old = E_mask
                    for seed in range(rp):
                        if E_mask[seed, 0] == 0:
                            Q = [(0, seed)]
                            break
                    if not Q:
                        for seed in range(rp):
                            if E_mask[0, seed] == 0:
                                Q=[(seed, 0)]
                                break
                    E_mask = fill_mask(E_mask, Q)

                    # Mask 대로 Overlap 부분 절단 mask의 1인 부분은 old, 0인 부분은 new
                    OV_cut = E_mask*OV_old + (1-E_mask)*OV_new
                    # HRDV에 붙여넣기
                    HR_DV[GY:GY+rp, GX:GX+OV_rp] = OV_cut
                    HR_DV[:rp, GX+OV_rp: GX+rp] = np.squeeze(HR_patch[:, OV_rp:])

                else: # Initial
                    HR_DV[:rp, :rp] = HR_patch

        plt.imsave("%sQuilt_subject%d.png"%(filepath, subject), HR_DV, origin='lower')