'''

data: https://download.pytorch.org/tutorial/hymenoptera_data.zip
Pytorch 0.4.0
tutorials: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''


from __future__ import print_function, division
import os
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import time
from torchvision import datasets, models
import torchvision.transforms as tf
import torch.utils.data as data
import matplotlib.pyplot as plt
import copy



# Hyper Parameters
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# #################################### data ##########################################################

train_transform = tf.Compose([tf.RandomResizedCrop(224),
                              tf.RandomHorizontalFlip(),
                              tf.ToTensor(),
                              tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transform = tf.Compose([tf.Resize(256),
                            tf.CenterCrop(224),
                            tf.ToTensor(),
                            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder('./hymenoptera_data/train', train_transform)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.ImageFolder('./hymenoptera_data/val', val_transform)
val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

data_loaders = {'train': val_loader, 'val':val_loader}
class_names = train_dataset.classes
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# #####################################################################################################


# ################################## imshow function ##################################################

def imshow(img, title=None):
    '''
    show image for tensor
    :param img:image
    :param title: title
    :return:
    '''
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# #####################################################################################################


# ################################### show some images ################################################

def show_example():
    inputs, classes = next(iter(train_loader))
    print(inputs.size())
    outpus = torchvision.utils.make_grid(inputs)
    imshow(outpus, title=[class_names[x] for x in classes])
    plt.show()

# #####################################################################################################


# ############################################## train ################################################

def train_model(model, loss_func, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(DEVICE)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 15)

        # each epoch has train and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # training mode
            else:
                model.train(False)  # evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            for d in data_loaders[phase]:
                inputs, labels = d

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # forward
                outputs = model(inputs)
                predict = torch.max(outputs, 1)[1]
                loss = loss_func(outputs, labels)

                optimizer.zero_grad()

                # train mode, backward and update parameters
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # calculate the loss and correct
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predict == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load the best weights to the model

    return model

# #####################################################################################################


# ################################## show the performance of model ####################################

def visualize_model(model, num_images=6):
    images_counter = 0
    plt.figure()
    model.to(DEVICE)

    for i, d in enumerate(data_loaders['val']):
        inputs, _ = d

        inputs = inputs.to(DEVICE)

        outputs = model(inputs)
        predict = torch.max(outputs.data, 1)[1]

        for j in range(inputs.size()[0]):
            images_counter += 1
            ax = plt.subplot(num_images / 2, 2, images_counter)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[predict[j]]))
            imshow(inputs.cpu().data[j])

            if images_counter == num_images:
                return

# #####################################################################################################


# ################################## CNN Mode #########################################################
'''
加载一个预训练的网络, 并重置最后一个全连接层.
'''
def train1():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    loss_func = nn.CrossEntropyLoss()

    # all parameters have been optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)

    # 每 7 个迭代, 让 LR 衰减 0.1 因素
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, loss_func, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    torch.save(model_ft, 'model_ft.pkl')
    print('train1 finished!')

# #####################################################################################################


# ################################## CNN Mode #########################################################

def train2():
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    loss_func = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=1e-3, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, loss_func, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    torch.save(model_conv, 'model_conv.pkl')

# #####################################################################################################


train1()
model_ft = torch.load('model_ft.pkl')
visualize_model(model_ft)
plt.show()

train2()
model_conv = torch.load('model_conv.pkl')
visualize_model(model_conv)
plt.show()



