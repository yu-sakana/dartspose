import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 9, 5

filename = 'data.csv'
df= pd.read_csv(filename)

for i in range(19):
    df1 = df[(df['human'] ==1) & (df['point'] == i)]
    plt.plot(df1['x'],df1['y'], label=i,marker = 'o')
plt.legend()
plt.show()

# for i in range(19):
#     df1 = df[(df['human'] ==1) & (df['point'] == i)]
#     y = np.sqrt((df1['x'])**2 + (df1['y'])**2)
#     plt.scatter(df1['frame']/30, y, label=i)
# plt.show()
