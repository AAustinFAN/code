import matplotlib as plt
import pandas as pd
import  pickle

chillers = pd.read_csv('data/chiller.csv',usecols=[5,10,15,20],encoding='UTF-8')
chillers.columns = ['chiller1_coolingLoad','chiller2_coolingLoad','chiller3_coolingLoad','chiller4_coolingLoad',]
chillers = chillers.multiply(24)

pd.set_option('display.max_columns', None)
with open('coolingLoad_1-4.pkl', 'wb') as f:
    pickle.dump(chillers, f)
    f.close()
print(chillers)


chillers['Col_sum'] = chillers.apply(lambda x: x.sum(), axis=1)
chillers.to_csv('Huarun_CoolingLoad_each_and_sum.csv')
