import pandas as pd
import numpy as np
from pylab import rcParams
import glob
from natsort import natsorted
import re
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import datetime
import os

def dir_check(ta):
    if not os.path.exists('ticc/data/{}'.format(ta)):
        os.mkdir('ticc/data/{}/'.format(ta))
    if not os.path.exists('image/{}/'.format(ta)):
        os.mkdir('image/{}/'.format(ta))

def Normalization(p):
  min_p = p.min()
  max_p = p.max()
  nor = (p - min_p) / (max_p - min_p)
  return nor

def out_rad(now,now_b,filename):
    df= pd.read_csv(filename)
    df1 = df[(df['human'] ==1) & (df['point'] == 2)]
    df2 = df[(df['human'] ==1) & (df['point'] == 3)]
    df3 = df[(df['human'] ==1) & (df['point'] == 4)]
    
    df1_x = df1['x']
    df1_y = df1['y']
    df2_x = df2['x']
    df2_y = df2['y']
    df3_x = df3['x']
    df3_y = df3['y']

    p1_x = df1_x.to_numpy()
    p1_y = df1_y.to_numpy()
    p2_x = df2_x.to_numpy()
    p2_y = df2_y.to_numpy()
    p3_x = df3_x.to_numpy()
    p3_y = df3_y.to_numpy()
    
    le_a = []
    le = []
    s = []
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
            le.append(a)
            s.append(j)

    le_np = np.array(le)
    window = 5
    w = np.ones(window)/window
#    x = np.convolve(le_np, w, mode='same')
    count = np.array(s)
    nor_a = Normalization(le_np)
    # plt.plot(s,x)
    # plt.savefig("image/{}/{}.jpeg".format(now,now_b))
    # plt.show()
    con = np.stack([count, nor_a],1)
#    print(con)
    with open('ticc/data/{}/{}_test.csv'.format(now,now_b),'w') as f:
        np.savetxt(f,con,delimiter=',')

if __name__ == '__main__':
#    import sys
#    sys.path.append('../')
    now = datetime.datetime.now()
    now_a = now.strftime('%Y_%m%d')
    now_b = now.strftime('%m%d_%H%M')
    dir_check(now_a)
    filename = 'csv/2021_1130/20211130_0046.csv'
#    filename = 'csv/2021_1126/20211126_1346.csv'
    out_rad(now_a,now_b,filename)
    
