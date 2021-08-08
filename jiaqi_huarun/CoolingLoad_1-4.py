import matplotlib as plt
import pandas as pd
import  pickle

chillers = pd.read_csv('data/chiller.csv',usecols=[5,10,15,20],encoding='gbk')
chillers.columns = ['chiller1_coolingLoad','chiller2_coolingLoad','chiller3_coolingLoad','chiller4_coolingLoad',]
chillers = chillers.multiply(24)
pd.set_option('display.max_columns', None)
with open('coolingLoad_1-4.pkl', 'wb') as f:
    pickle.dump(chillers, f)
    f.close()
print(chillers)

