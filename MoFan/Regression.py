import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import matplotlib.pyplot as plt

# data
# unsqueeze将一维数据转行成二维数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data(tensor),shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data(tensor), shape=(100, 1)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  #hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion()

for i in range(600):
    prediction = net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # bp, compute gradients
    optimizer.step()  # update

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
