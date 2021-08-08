# we need COP of each chiller
#COP 已经在之前的file中计算过
import pickle
import pandas as pd
import numpy as np


def export_cop_csv():
    with open('coplist.pkl', 'rb') as f:
        coplist = pickle.load(f)
        f.close()
    t = np.array(coplist)
    df = pd.DataFrame(t.transpose(), columns=['1', '2', '3', '4'])
    df.to_csv('4chillers_cop.csv', index=False)


def readdata():
    datax = pd.read_csv('../data/chiller.csv',usecols=[1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19])
    temp = pd.read_csv('../data/temperature(2).csv',usecols=[1,2])
    datax = pd.concat([datax,temp],axis=1)
    datax= datax.values
    # with open('coplist.pkl', 'rb') as f:
    #     coplist = pickle.load(f)
    #     f.close()
    # datay = coplist[2].reshape(4361,1)
    datay = pd.read_csv('../data/chiller.csv',usecols=[15]).values
    return datax,datay  # numpy type
readdata()


