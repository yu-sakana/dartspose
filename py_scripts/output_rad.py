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

now = datetime.datetime.now()
filename = 'csv/20211125_14_10.csv'
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
        l = [j,a]
        le.append(a)
        s.append(j)
        le_a.append(l)

plt.plot(s,le)
plt.savefig("img.png")
plt.show()
with open('ticc/data/{}.txt'.format(now.strftime('%Y%m%d_%H_%M')),'w') as f:
    np.savetxt(f,le_a,delimiter=',')
