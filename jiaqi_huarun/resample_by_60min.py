import pandas as pd
import pickle

#for 冷机功率
def got_electrical_power_1(path):
    print(path)
    chiller1_power = pd.read_excel('/Users/austin/PycharmProjects/华润/data/冷水主机运行功率/'+path,usecols=[0,2])
    chiller1_power['record_timestamp'] = pd.to_datetime(chiller1_power['日期/时间'], utc=True)
    chiller1_power = chiller1_power.set_index('record_timestamp')
    chiller1_power = chiller1_power.resample('60T').mean()
    chiller1_power.fillna(method='backfill', axis=0, inplace=True)
    with open('/Users/austin/PycharmProjects/华润/data/冷水主机运行功率/'+path+'.pkl', 'wb') as f:
        pickle.dump(chiller1_power, f)
        f.close()
    # pd.set_option('display.max_rows', None)
    print(chiller1_power)

pathlist = ['1号冷机/1号冷机功率（上半年）.xls','1号冷机/1号冷机功率（下半年）.xls','2号冷机/2号冷机功率（上半年）.xls','2号冷机/2号冷机功率（下半年）.xls',
            '3号冷机/3号冷机功率（上半年）.xls','3号冷机/3号冷机功率（下半年）.xls','4号冷机/4号冷机功率（上半年）.xls','4号冷机/4号冷机功率（下半年）.xls',]

for path in pathlist:
    got_electrical_power_1(path)