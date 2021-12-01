
import matplotlib.pyplot as plt
import pandas as pd

#f = open('Results.csv', 'r')

data = pd.read_csv('res/Results_1128_1600.csv')
# te = pd.read_csv('data/2021_1126/2021_1126_test.csv')
# print(te)
# te_x = te['frame']
# te_y = te['rad']
# te_res = te['res']
#for i in range(len(te_x)):
#    plt.plot(te_x[i:i+2], te_y[i:i+2], color= 'red' if te_res[i] == 0 else 'blue')
plt.plot(data)
#plt.savefig('test_plot_1316.jpg')
plt.show()
