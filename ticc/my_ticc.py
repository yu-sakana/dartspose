from TICC_solver import TICC
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import datetime


def main():
    day = '1128_1600'
    fname = "data/2021_1128/1128_1559_test.csv"
#    fname = 'data/2021_1126/1126_1349_test.csv'
    ticc = TICC(window_size=3, number_of_clusters=4, lambda_parameter=11e-2, beta=100, maxIters=10, threshold=2e-5,
            write_out_file=True, prefix_string="output_folder/", num_proc=1)
    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

    print(cluster_assignment)
    le = []
    l = []
    [le.append(i) for i in range(len(cluster_assignment))]
    np.savetxt('res/Results_{}.csv'.format(day), cluster_assignment,fmt='%d',delimiter=',')
    plt.plot(le,cluster_assignment)
    plt.savefig('ticc_image/ticc_{}.jpg'.format(day)) 
#    with open(fname,mode='a') as f:
#    df = pd.concat(le, axis=0, sort=True)

if __name__ == '__main__':
    main()
