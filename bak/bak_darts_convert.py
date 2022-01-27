import pandas as pd
import numpy as np
from pylab import rcParams
import glob
from natsort import natsorted
import re
from numpy import linalg as LA
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib.gridspec as gridspec
import seaborn as sns

def dir_check(now_time):
    if not os.path.exists('ticc/data/{}'.format(now_time)):
        os.mkdir('ticc/data/{}/'.format(now_time))
    if not os.path.exists('image/{}/'.format(now_time)):
        os.mkdir('image/{}/'.format(now_time))

def convert_rad(df):
    df1 = df[(df['human'] ==1) & (df['point'] == 2)]
    df2 = df[(df['human'] ==1) & (df['point'] == 3)]
    df3 = df[(df['human'] ==1) & (df['point'] == 4)]
    
    df1_x = df1['x'];df1_y = df1['y']
    df2_x = df2['x'];df2_y = df2['y']
    df3_x = df3['x'];df3_y = df3['y']

    p1_x = df1_x.to_numpy();p1_y = df1_y.to_numpy()
    p2_x = df2_x.to_numpy();p2_y = df2_y.to_numpy()
    p3_x = df3_x.to_numpy();p3_y = df3_y.to_numpy()
    
    rad_list = [];frame_count = []
    for j in range(len(p3_x)):
        u = np.array([p1_x[j] - p2_x[j], p1_y[j] - p2_y[j]])
        v = np.array([p3_x[j] - p2_x[j], p3_y[j] - p2_y[j]])
        i = np.inner(u, v)

        n = LA.norm(u) * LA.norm(v)
        if n == 0:
            a = 0
        else:
            c = i / n
            a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
            rad_list.append(a)
            frame_count.append(j)

    return frame_count,rad_list

def normalization(p):
    min_p = p.min()
    max_p = p.max()
    nor = (p - min_p) / (max_p - min_p)
    return nor

def rad_convert_nor(rad_list):
    rad = np.array(rad_list)
#    count = np.array(frame_count)
    nor_list = normalization(rad)
#    con_list = np.stack([count, nor_list],1)
    return nor_list

def save_dataframe(rad_list,con_list):
    df = pd.DataFrame({'frame':con_list[:,0],'rad':con_list[:,1],'nor_rad':rad_list[:,0]})
    print(df)
    return df
    
d = lambda a,b: (a - b)**2
first = lambda x: x[0]
second = lambda x: x[1]

def minVal(v1, v2, v3):
    if first(v1) <= min(first(v2), first(v3)):
        return v1, 0
    elif first(v2) <= first(v3):
        return v2, 1
    else:
        return v3, 2 

def calc_dtw(A,B):
    S = len(A)
    T = len(B)

    m = [[0 for j in range(T)] for i in range(S)]
    m[0][0] = (d(A[0],B[0]), (-1,-1))
    for i in range(1,S):
        m[i][0] = (m[i-1][0][0] + d(A[i], B[0]), (i-1,0))
    for j in range(1,T):
        m[0][j] = (m[0][j-1][0] + d(A[0], B[j]), (0,j-1))

    for i in range(1,S):
        for j in range(1,T):
            minimum, index = minVal(m[i-1][j], m[i][j-1], m[i-1][j-1])
            indexes = [(i-1,j), (i,j-1), (i-1,j-1)]
            m[i][j] = (first(minimum)+d(A[i], B[j]), indexes[index])
    return m

def backward(m):
    path = []
    path.append([len(m)-1, len(m[0])-1])
    while True:
        path.append(m[path[-1][0]][path[-1][1]][1])
        if path[-1]==(0,0):
            break
    path = np.array(path)
    return path

def plot_path(m,path, A, B):
    gs = gridspec.GridSpec(2, 2,width_ratios=[1,5],height_ratios=[5,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])
    
    list_d = [[t[0] for t in row] for row in m]
    list_d = np.array(list_d)
    ax2.pcolor(list_d, cmap=plt.cm.Blues)
    ax2.plot(path[:,1], path[:,0], c="C3")
    
    ax1.plot(A, range(len(A)))
    ax1.invert_xaxis()
    ax4.plot(B, c="C1")
    plt.show()
    
    for line in path:
        plt.plot(line, [A[line[0]], B[line[1]]], linewidth=0.2, c="gray")
    plt.plot(A,label='model throw')
    plt.plot(B,label='your throw')
    plt.show()
#    plt.savefig('test.jpeg')

# def plot_path(paths, A, B, D):
#     plt.figure(figsize=(5,5))
#     gs = gridspec.GridSpec(2, 2,
#                        width_ratios=[1,5],
#                        height_ratios=[5,1]
#                        )
#     ax1 = plt.subplot(gs[0])
#     ax2 = plt.subplot(gs[1])
#     ax4 = plt.subplot(gs[3])

#     ax2.pcolor(D, cmap=plt.cm.Blues)
#     ax2.get_xaxis().set_ticks([])
#     ax2.get_yaxis().set_ticks([])
    
#     for path in paths:
#         ax2.plot(path[:,0]+0.5, path[:,1]+0.5, c="C3")
    
#     ax4.plot(A)
#     ax4.set_xlabel("$X$")
#     ax1.invert_xaxis()
#     ax1.plot(B, range(len(B)), c="C1")
#     ax1.set_ylabel("$Y$")

#     ax2.set_xlim(0, len(A))
#     ax2.set_ylim(0, len(B))
#     plt.show()

# def dist(x, y):
#     return (x - y)**2

# def get_min(m0, m1, m2, i, j):
#     if m0 < m1:
#         if m0 < m2:
#             return i-1, j, m0
#         else:
#             return i-1, j-1, m2
#     else:
#         if m1 < m2:
#             return i, j-1, m1
#         else:
#             return i-1, j-1, m2
        
# def partial_dtw(x, y):
#     Tx = len(x)
#     Ty = len(y)

#     C = np.zeros((Tx, Ty))
#     B = np.zeros((Tx, Ty, 2), int)

#     C[0, 0] = dist(x[0], y[0])
#     for i in range(Tx):
#         C[i, 0] = dist(x[i], y[0])
#         B[i, 0] = [0, 0]

#     for j in range(1, Ty):
#         C[0, j] = C[0, j - 1] + dist(x[0], y[j])
#         B[0, j] = [0, j - 1]

#     for i in range(1, Tx):
#         for j in range(1, Ty):
#             pi, pj, m = get_min(C[i - 1, j],
#                                 C[i, j - 1],
#                                 C[i - 1, j - 1],
#                                 i, j)
#             C[i, j] = dist(x[i], y[j]) + m
#             B[i, j] = [pi, pj]
#     t_end = np.argmin(C[:,-1])
#     cost = C[t_end, -1]
    
#     path = [[t_end, Ty - 1]]
#     i = t_end
#     j = Ty - 1

#     while (B[i, j][0] != 0 or B[i, j][1] != 0):
#         path.append(B[i, j])
#         i, j = B[i, j].astype(int)
        
#     return np.array(path), cost
