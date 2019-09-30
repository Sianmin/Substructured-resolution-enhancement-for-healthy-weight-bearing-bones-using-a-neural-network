import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    [ratio, subject] = [10, 1]
    [height, width] = [2080, 1883]
    [NY, NX] = [math.floor(height/ratio), math.floor(width/ratio)]

    Displacement = np.zeros((NY+1, NX+1, 6))

    for load_case in range(1,4):
        dis_file = open("r%d\s%d\Displacement_r%d_s%d_c%d" % (ratio, subject, ratio, subject, load_case), 'r')
        while 1:
            lines = dis_file.readline().split()
            if not lines: break
            [node_x, node_y, dx, dy] = [int(eval(lines[0])), int(eval(lines[1])), eval(lines[2]), eval(lines[3])]
            Displacement[node_y, node_x, (load_case - 1) * 2:load_case * 2] = [dx, dy]
        dis_file.close()

        slicing = 5
        plt.subplot(1, 3, load_case); plt.xlim(-1, NX+1); plt.ylim(-1, NY+1); plt.axis('equal')
        [X, Y] = np.meshgrid(np.arange(0, NX+1, slicing), np.arange(0, NY+1, slicing))
        Dis= Displacement[0:NY+1:slicing, 0:NY+1:slicing, (load_case-1)*2:(load_case)*2]
        plt.quiver(X, Y, Dis[:,:,0], Dis[:,:,1], scale = 2)
    plt.show()