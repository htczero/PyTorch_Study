import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Hyper Paramters
DOWNLOAD_STATE = False
BATCH_SIZE = 4
EPOCH = 5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# training set
trainSet = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=DOWNLOAD_STATE, transform=transform)
trainLoader = data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)

# test set
testSet = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform)
testLoader = data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)

# labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ################## show image ########################################


def imshow(img):
    '''
    show image
    :param img: image
    :return:
    '''
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataIter = iter(trainLoader)
# images, labels = dataIter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

# #########################################################################


# ############################## CNN Module ###############################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# ###########################################################################


# ######################## train #############################################
def train():
    net = Net().cuda()

    # Loss and Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)

    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))

    print('Finished Training')
    torch.save(net, 'CNN_CIFAR10.pkl')

# ###########################################################################


# ########################### test #############################################


def testCNN():
    net = torch.load('CNN_CIFAR10.pkl')
    net = net.cpu()
    dataIter = iter(testLoader)
    images, labels = dataIter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
    with torch.no_grad():
        outputs = net(images)
        predict = torch.max(outputs, 1)[1].data.squeeze()
        print('Predicted: ', ' '.join('%5s' % classes[predict[j]]
                                      for j in range(BATCH_SIZE)))
    plt.show()

# ###########################################################################


# ########################## evaluate ############################################


def evaluate():
    net = torch.load('CNN_CIFAR10.pkl')
    net = net.cpu()
    correct = list(0. for i in range(10))
    total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            predict = torch.max(outputs, 1)[1].data.squeeze()
            c = (torch.eq(predict, labels).squeeze()).numpy()
            for i in range(BATCH_SIZE):
                label = labels[i]
                correct[label] += c[i]
                total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * correct[i] / total[i]))


# ###########################################################################


# train()
# testCNN()
evaluate()



