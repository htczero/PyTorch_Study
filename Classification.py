import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)   # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)   # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer, label 默认需要long


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.output = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.output(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for i in range(100):
    out = net(x)  # input x and predict based on x
    loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
    print(out)
    print(torch.max(out, 1)[1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        plt.cla()
        # torch.max(input, dim=1) 返回最大元素在这一行的列索引
        prediction = torch.max(F.softmax(out), 1)[1]  # [0]表示最大值，[1]表示最大值的索引值
        pred_y = y.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
