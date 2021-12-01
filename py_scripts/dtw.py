import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd

d = lambda a,b: (a - b)**2
first = lambda x: x[0]
second = lambda x: x[1]

def Normalization(p):
  min_p = p.min()
  max_p = p.max()
  nor = (p - min_p) / (max_p - min_p)
  return nor

def minVal(v1, v2, v3):
    if first(v1) <= min(first(v2), first(v3)):
        return v1, 0
    elif first(v2) <= first(v3):
        return v2, 1
    else:
        return v3, 2 

def calc_dtw(A, B):
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

import matplotlib.gridspec as gridspec

def plot_path(path, A, B):
    gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,5],
                       height_ratios=[5,1]
                       )
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
    plt.plot(A)
    plt.plot(B)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('test.csv',header=None)
#    o = df[0]
    o = np.array(df[0])
#    s = df[1]
    s = np.array(df[1])
    s = s[~np.isnan(s)]
    o = Normalization(o)
    s = Normalization(s)
    #print(s)
    #print(df)
    #m = calc_dtw(o,s)
    #print("your score: ",100*(1-calc_dtw(o, s)[-1][-1][0]))
    print(calc_dtw(o, s)[-1][-1][0])
    m = calc_dtw(o,s)
    path = backward(m)
    plot_path(path, o, s)
