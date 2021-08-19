import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import preprocess

input_dim = config.input_dim
hidden_size = config.hidden_size
batch_size = config.batch_size
train_ratio = config.train_ratio
learningRate = config.learningRate
epoch = config.epoch


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=1)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 1, bias=1)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

class TrainSet(Dataset):
    def __init__(self, datax, datay):
        self.data, self.label = datax, datay

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


# we need COP of each chiller
#COP 已经在之前的file中计算过
import pickle

datax,datay = preprocess.readdata()


print('data size is ', datax.shape, datay.shape)


#######
# trainset = TrainSet(datax[805:1453],datay[805:1453])
# 2021年2月3日 10:00 AM to 2021年3月2日11：00 AM
# trainset = TrainSet(datax[0:306],datay[0:306])
# 2021年1月1日 00:00 AM to 2021年1月13日 16：00 PM
# trainset = TrainSet(datax[3354:3528], datay[3354:3528])
# 2021年5月20日 16:00:00 PM to 2021年5月27日 22：00PM
########

#chiller 3
#2021.1.14 14:00 PM to 2021.2.3 11:00AM
# trainset = TrainSet(datax[328:805], datay[328:805])
trainset = TrainSet(datax[1454:1792], datay[1454:1792])



#####
# testset = TrainSet(datax[3354:3528], datay[3354:3528])
# 2021年5月20日 16:00 PM to 2021年5月27日 22：00PM

# testset = TrainSet(datax[4051:4362], datay[4051:4362])
# 2021年6月18日 16:00 PM to 2021年7月1日 16:00 PM

# testset = TrainSet(datax[805:1453],datay[805:1453])
# 2021年2月3日 10:00 AM to 2021年3月2日11：00 AM

# testset = TrainSet(datax[0:306],datay[0:306])
# 2021年1月1日 00:00 AM to 2021年1月13日 16：00 PM
#######

#chiller 3
#2021.3.2 12:00 PM to 2021.3.16 14:00AM
# testset = TrainSet(datax[1454:1792], datay[1454:1792])
#2021.4.26 10:00 AM to 2021.5.20 15:00 PM
testset = TrainSet(datax[2772:3353], datay[2772:3353])


ITER = 20
test_batch = 24

loss_result_all = np.zeros((ITER, testset.data.shape[0]//test_batch+1))
for II in range(ITER):





    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    Seq_model = Net(n_feature=input_dim, n_hidden=hidden_size)  # define the network
    optimizer = torch.optim.SGD(Seq_model.parameters(), lr=learningRate)
    loss_func = nn.L1Loss()

    for step in range(epoch):
        Loss_list = []
        prelist = []
        for x, y in trainloader:
            x = x.to(torch.float32)
            y = y.to(torch.float32)

            prediction = Seq_model(x)  # input x and predict based on x
            for x in prediction:
                prelist.append(x.detach().numpy())
            # print(prediction,y)
            loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
            Loss_list.append(loss.item())
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # print('Epoch:{}, Loss:{:.5f}'.format(step + 1, loss.item()))

    # x = np.linspace(0, len(prelist), len(prelist))
    # x = np.linspace(0, len(Loss_list), len(Loss_list))
    # plt.plot(range(Loss_list.__len__()),Loss_list)
    # plt.plot(x, prelist)
    # plt.plot(x, trainset.label)
    # plt.legend(labels = ['prelist','label'])
    # plt.title("train loss show")
    # plt.show()







    testloader = DataLoader(testset, batch_size=test_batch, shuffle=False)

    loss_func = nn.L1Loss()
    Loss_list = []
    prelist = []

    for x, y in testloader:
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        prediction = Seq_model(x)  # input x and predict based on x
        for x in prediction:
            prelist.append(x.detach().numpy())
        # print(prediction,y)
        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
        Loss_list.append(loss.item())
        # print('Epoch:{}, Loss:{:.5f}'.format(1, np.mean(Loss_list)))

        # # todo update
        # optimizer.zero_grad()  # clear gradients for next train
        # loss.backward()  # backpropagation, compute gradients
        # optimizer.step()  # apply gradients





    # x = np.linspace(0, len(Loss_list), len(Loss_list))
    # plt.plot(x, Loss_list)
    # plt.show()

    # x = np.linspace(0, len(prelist), len(prelist))
    # plt.plot(x, prelist)
    # plt.plot(x, testset.label, 'r')
    # l = np.linspace(0, len(Loss_list), len(Loss_list))
    # plt.plot(range(Loss_list.__len__()),Loss_list)
    # plt.title("inference, without update")
    # plt.show()

    loss_result_all[II, :] = np.array(Loss_list)


mean_iter_loss = np.mean(loss_result_all, axis=0)
print(mean_iter_loss)
plt.scatter(range(mean_iter_loss.__len__()), mean_iter_loss)
plt.title("inference round= %s, var=%s, std=%s" % (ITER, loss_result_all.var(), loss_result_all.std()))
plt.show()