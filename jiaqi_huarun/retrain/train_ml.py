import numpy as np

import preprocess
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


datax, datay = preprocess.readdata()

print('data size is ', datax.shape, datay.shape)

x_train = datax[0:306]
y_train = datay[0:306]
x_test =  datax[805:1453]
y_test = datay[805:1453]

# todo update
x_train = datax[0:306]
x_train = np.append(x_train, datax[805:829],axis=0)
y_train = datay[0:306]
y_train = np.append(y_train,datay[805:829],axis=0)
x_test =  datax[829:1453]
y_test = datay[829:1453]


poly_svr = SVR(kernel='poly')  # 多项式核函数初始化的SVR
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)

import matplotlib.pyplot as plt
loss_list= []
# plt.plot(range(poly_svr_y_predict.__len__()),poly_svr_y_predict)
# plt.show()
for i in range(len(y_test)):
    loss = y_test[i]-poly_svr_y_predict[i]
    loss_list.append(loss)

plt.plot(range(loss_list.__len__()),loss_list)
plt.show()

