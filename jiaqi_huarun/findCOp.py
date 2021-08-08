import pickle

import matplotlib.pyplot as plt
import numpy as np

with open('coolingLoad_1-4.pkl', 'rb') as f:
    CL = pickle.load(f)
    f.close()
columnlist = ['chiller1_coolingLoad', 'chiller2_coolingLoad', 'chiller3_coolingLoad',
       'chiller4_coolingLoad']
print(CL)

pathlist1 = ['1号冷机/1号冷机功率（上半年）.xls.pkl','2号冷机/2号冷机功率（上半年）.xls.pkl','3号冷机/3号冷机功率（上半年）.xls.pkl','4号冷机/4号冷机功率（上半年）.xls.pkl']
pathlist2 = ['1号冷机/1号冷机功率（下半年）.xls.pkl','2号冷机/2号冷机功率（下半年）.xls.pkl','3号冷机/3号冷机功率（下半年）.xls.pkl','4号冷机/4号冷机功率（下半年）.xls.pkl']
coplist = []
powerlist = []
for i in range(4):
    with open(pathlist1[i], 'rb') as f:
        power = pickle.load(f)
        f.close()
    power1 = power.tail(288)

    with open(pathlist2[i], 'rb') as f:
        power = pickle.load(f)
        f.close()
    power2 = power.head(4073)
    power = power1.append(power2)
    n1 = power.values.flatten()
    n2 = CL[columnlist[i]].values
    cop = n2/n1
    print(cop)
    coplist.append(cop)
    powerlist.append(n1)
    with open('/Users/austin/PycharmProjects/华润/retrain/coplist.pkl', 'wb') as f:
        pickle.dump(coplist, f)
        f.close()
print(coplist)
# for cop in coplist:
#     cop= cop[0:4361:24]
#     x = np.linspace(0, len(cop), len(cop))
#     plt.plot(x, cop)
#     plt.legend(labels=['chiller1','chiller2','chiller3','chiller4'],loc='upper right')
#     plt.xlabel('time from 2021.1.1 to 2021.7.16')
#     plt.ylabel('cop')
# plt.show()

for power in powerlist:
    power = power[0:4361:24]
    x = np.linspace(0, len(power), len(power))
    plt.plot(x, power)
    plt.legend(labels=['chiller1', 'chiller2', 'chiller3', 'chiller4'], loc='upper right')
plt.show()