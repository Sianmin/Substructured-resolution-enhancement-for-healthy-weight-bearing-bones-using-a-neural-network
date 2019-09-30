import numpy as np
import math

[height, width] = [2080, 1883]
case_n = 3
resolution = 0.05

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def readESB(name, subject, metric): # read Elemental Structural Behavior
    vals = np.zeros((height, width, case_n))
    for load_case in range(1, case_n+1):
        with open("FEA/%s/%s_s%d_c%d" % (name, metric, subject, load_case), 'r') as ref_fid:
            while 1:
                lines = ref_fid.readlines(100000)
                if not lines: break
                for line in lines:
                    parse = line.split()
                    [elem_x, elem_y, val] = [math.floor(eval(parse[0])/resolution), math.floor(eval(parse[1])/resolution), eval(parse[2])]
                    vals[elem_y, elem_x, load_case-1] = val
    return vals

def readNSB(name, subject, metric): # read Nodal Structural Behavior
    vals = np.zeros((height + 1, width + 1, case_n))
    for load_case in range(1, case_n+1):
        with open("FEA/%s/%s_c%d" % (name, metric, load_case), 'r') as ref_fid:
            while 1:
                lines = ref_fid.readlines(100000)
                if not lines: break
                for line in lines:
                    parse = line.split()
                    [elem_x, elem_y, val] = [math.floor(eval(parse[0])/resolution), math.floor(eval(parse[1])/resolution), eval(lines[2])]
                    vals[elem_y, elem_x, load_case-1] = val
    return vals