# -*- coding:utf-8 -*-
'''
pytorch 0.4.0
data: https://download.pytorch.org/tutorial/faces.zip
tutorials: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''


from __future__ import print_function, division
import os
import torch
import pandas as pd  # 用于解析csv文件
from skimage import io, transform  # 用于读取图片和图像变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

plt.ion()  # 交互模式

# ########################## 加载csv文件 ##################################################


def load_csv(path):
    '''
    加载csv文件
    :param path: csv文件的路径
    :return: 图片名字， 标记点
    '''
    landmarks_frame = pd.read_csv(path)

    n = 65
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))

    return img_name, landmarks

# #########################################################################################


# ############################### 显示图片 #################################################

def show_landmarks(image, landmarks):
    '''
    显示带有标记点的图片
    :param image: 图片
    :param landmarks:标记点
    :return:
    '''

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)


# plt.figure()
# img_name, landmarks = load_csv('./faces/face_landmarks.csv')
# show_landmarks(io.imread(os.path.join('faces/', img_name)), landmarks)
# plt.show()

# #########################################################################################


# ############################### 显示图片 #################################################

'''
torch.utils.data.Dataset 是一个表示数据集的抽象类. 你自己的数据集一般应该继承``Dataset``, 并且重写下面的方法:

__len__ 使用``len(dataset)`` 可以返回数据集的大小
__getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第:math:i个样本(0索引)
'''


class FaceLandmarksDataset(Dataset):
    '''
    人脸标记数据集
    '''
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        :param csv_file: 带有标记的CSV文件
        :param root_dir:图片路径
        :param transform:可以选择的图像变换
        '''
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# ############################################################################################


# ############################### 改变图像大小 #################################################
'''
大多数神经网络需要输入 一个固定大小的图像, 因此我们需要写代码来处理. 让我们创建三个transform操作:

Rescale: 修改图片尺寸
RandomCrop: 随机裁切图片, 这是数据增强的方法
ToTensor: 将numpy格式的图片转为torch格式的图片（我们需要交换坐标轴）
我们不将它们写成简单的函数, 而是写成可以调用的类, 这样transform的参数不需要每次都传递 如果需要的话, 我们只
需实现 __call__ 方法和``__init__`` 方法.之后我们可以像下面这 样使用transform:

tsfm = Transform(params)
transformed_sample = tsfm(sample)
'''


class Rescale(object):

    def __init__(self, output_size):
        '''
        Args:
            output_size (tuple or int): 要求输出的尺寸.  如果是个元组类型, 输出
            和output_size匹配. 如果时int类型,图片的短边和output_size匹配, 图片的
            长宽比保持不变.
        :param output_size:
        :return:
        '''
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]  # shaple=(hight, weight, channels)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # 对于标记点, h和w需要交换位置, 因为对于图像, x和y分别时第1维和第0维
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

# ############################################################################################


# ############################### 随机裁剪图片 #################################################

class RandomCrop(object):
    '''
    Args:
        output_size (tuple or int): 期望输出的尺寸, 如果时int类型, 裁切成正方形.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

# ############################################################################################


# ############################### 数据类型转变，ndarrays->Tensor ###############################

class ToTensor(object):
    '''
    ndarrays->Tensor
    '''
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # numpy 图像：H, W, C
        # torch 图像：C, H, W
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        landmarks = torch.from_numpy(landmarks)

        return {'image': image, 'landmarks': landmarks}

# ############################################################################################


# ############################### Compose transforms and DataLoader ##########################
'''
如果我们想将图片的短边变为256像素, 并且随后随机裁切成224像素的正方形. i.e, 
``Rescale``和``RandomCrop``变换. torchvision.transforms.Compose 
就是一个可以做这样一个组合的可调用的类.

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''


def get_data_loader(path, root_path):
    transformed_dataset = FaceLandmarksDataset(csv_file=path,
                                               root_dir=root_path,
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))
    data_loader = DataLoader(transformed_dataset, batch_size=4, shuffle=True)

    return data_loader

# ############################################################################################


# ############################### 显示指定 batch 的数据样本的图片和标记点 #########################

def show_landmarks_batch(sample_batched):

    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    gird = utils.make_grid(images_batch)
    plt.imshow(gird.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')

# ############################################################################################


data_loader = get_data_loader('./faces/face_landmarks.csv','./faces')

for i_batch, sample_batched in enumerate(data_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # 观察到第四批数据时停止
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
