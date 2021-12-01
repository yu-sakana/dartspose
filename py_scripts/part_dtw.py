import numpy as np
import pylab as plt
import pandas as pd
import matplotlib.gridspec as gridspec

def normalization(p):
  min_p = p.min()
  max_p = p.max()
  nor = (p - min_p) / (max_p - min_p)
  return nor

def plot_path(paths, A, B, D):
    plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,5],
                       height_ratios=[5,1]
                       )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])

    ax2.pcolor(D, cmap=plt.cm.Blues)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    [ax2.plot(path[:,0]+0.5, path[:,1]+0.5, c="C3") for path in paths]
    
#    for path in paths:
#        ax2.plot(path[:,0]+0.5, path[:,1]+0.5, c="C3")
    
    ax4.plot(A)
    ax4.set_xlabel("$X$")
    ax1.invert_xaxis()
    ax1.plot(B, range(len(B)), c="C1")
    ax1.set_ylabel("$Y$")

    ax2.set_xlim(0, len(A))
    ax2.set_ylim(0, len(B))
    plt.show()
    
def dist(x, y):
    return (x - y)**2

def get_min(m0, m1, m2, i, j):
    if m0 < m1:
        if m0 < m2:
            return i - 1, j, m0
        else:
            return i - 1, j - 1, m2
    else:
        if m1 < m2:
            return i, j - 1, m1
        else:
            return i - 1, j - 1, m2

def partial_dtw(x, y):
    Tx = len(x)
    Ty = len(y)

    C = np.zeros((Tx, Ty))
    B = np.zeros((Tx, Ty, 2), int)

    C[0, 0] = dist(x[0], y[0])
    for i in range(Tx):
        C[i, 0] = dist(x[i], y[0])
        B[i, 0] = [0, 0]

    for j in range(1, Ty):
        C[0, j] = C[0, j - 1] + dist(x[0], y[j])
        B[0, j] = [0, j - 1]

    for i in range(1, Tx):
        for j in range(1, Ty):
            pi, pj, m = get_min(C[i - 1, j],
                                C[i, j - 1],
                                C[i - 1, j - 1],
                                i, j)
            C[i, j] = dist(x[i], y[j]) + m
            B[i, j] = [pi, pj]
    t_end = np.argmin(C[:,-1])
    cost = C[t_end, -1]
    
    path = [[t_end, Ty - 1]]
    i = t_end
    j = Ty - 1

    while (B[i, j][0] != 0 or B[i, j][1] != 0):
        path.append(B[i, j])
        i, j = B[i, j].astype(int)
        
    return np.array(path), cost

if __name__ == '__main__':
    df = pd.read_csv('test.csv',header=None)
    #X:test,Y:train
    X = df[0]
    Y = df[1]
    Y = Y[~np.isnan(Y)]
#    print(X,Y)
    Y = normalization(Y)
    X = normalization(X)
    path, cost = partial_dtw(X, Y)
    print(cost)
    
    D = (np.array(X).reshape(1, -1) - np.array(Y).reshape(-1, 1))**2
#    plot_path([np.array(path)], X, Y, D)

    for line in path:
        plt.plot(line, [X[line[0]], Y[line[1]]], linewidth=0.8, c="gray")
    plt.plot(X)
    plt.plot(Y)
    plt.plot(path[:,0], X[path[:,0]], c="C2")
    plt.show()
