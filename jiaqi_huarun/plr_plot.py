import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plr = pd.read_excel('plr_1-4.xlsx')
heads = ['HR_SYS/冷水机1实际负荷 (%)', 'HR_SYS/冷水机2实际负荷 (%)', 'HR_SYS/冷水机3实际负荷 (%)',
       'HR_SYS/冷水机4实际负荷 (%)']
chiller1 = plr[heads[0]].values
chiller1 = chiller1[chiller1 != 0]
chiller2 = plr[heads[1]].values
chiller2 = chiller2[chiller2 != 0]

chiller3 = plr[heads[2]].values
chiller3 = chiller3[chiller3 != 0]

chiller4 = plr[heads[3]].values
chiller4 = chiller4[chiller4 != 0]


x = np.linspace(0, len(chiller1), len(chiller1))
# plt.hist(chiller1, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.show()
# plt.hist(chiller2, bins=40, normed=0, facecolor="red", edgecolor="black", alpha=0.7)
# plt.show()
# plt.hist(chiller3, bins=40, normed=0, facecolor="green", edgecolor="black", alpha=0.7)
# plt.show()
plt.hist(chiller4, bins=40, normed=0, facecolor="black", edgecolor="black", alpha=0.7)
plt.show()

# # plt.hist(x,chiller1)
# plt.legend(labels=['chiller1','chiller2','chiller3','chiller4'],loc='upper right')
# plt.xlabel('time from 2021.1.1 to 2021.7.16')
# plt.ylabel('cop')
# plt.show()