import os
import math
from pickletools import optimize
from random import shuffle
import time

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data


Age2label = {'(0, 2)': 0, '(4, 6)': 1, '(8, 13)': 2, '(15, 20)': 3, '(25, 32)': 4, '(38, 43)': 5, '(48, 53)': 6, '(60, 100)': 7}
Gender2label = {"m": 0, "f": 1}
Epoches = 2
Batch_size = 50
lr = 0.0005


class CNN(nn.Module):
    def __init__(self, num_class, num_in_channels=3):
        super().__init__()

        # conv by 96 7x7 filters + maxpool + LRN + ReLU
        self.cnnblock1 = nn.Sequential(
            nn.Conv2d(in_channels=num_in_channels, out_channels=96, 
                      kernel_size=7, stride=4, padding=3),
                # input_channels, output_channels, kernel_size, padding
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.ReLU()
        )   # output 96x28x28

        # conv by 256 5x5 filters + maxpool + LRN + ReLU
        self.cnnblock2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, 
                      kernel_size=5, padding=3),
                # input_channels, output_channels, kernel_size, padding
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),
            nn.ReLU()
        )   # output 256x14x14

        # conv by 384 3x3 filters + maxpool + ReLU (NO LRN this time)
        self.cnnblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, 
                      kernel_size=3, padding=1),
                # input_channels, output_channels, kernel_size, padding
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )   # output 384x6x6

        # fully connected layer 1: 512 neurons + dropout=0.5
        self.linearblock1 = nn.Sequential(
            nn.Linear(in_features=384*6*6, out_features=512, bias=False),
            nn.Dropout(),
            nn.ReLU()
        )

        self.linearblock2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=False),
            nn.Dropout(),
            nn.ReLU()
        )

        self.linearclass = nn.Linear(in_features=512, out_features=num_class, bias=False)
        self.softmax = nn.Softmax(dim=-1)       


    def forward(self, x):
        x = self.cnnblock1(x)           # 96*28*28
        x = self.cnnblock2(x)           # 256*14*14
        x = self.cnnblock3(x)           # 
        x = x.view(x.size(0), -1)       # flatten x as batch, -1 for linear
        x = self.linearblock1(x)        # 512
        x = self.linearblock2(x)        # 512
        x = self.linearclass(x)         # num_class (depend on age/gender)
        x = self.softmax(x)
        return x


def np2tensor(age_array, gender_array):
    """
    Return two arrays of torch tensor of age_array and gender_array.
    """
    ages, genders = [], []
    for i in range(age_array.shape[0]):
        ages.append(Age2label[age_array[i]])
        genders.append(Gender2label[gender_array[i]])

    return torch.LongTensor(ages), torch.LongTensor(genders)


def truth2label(truth_array, num_class):
    """
    Return the one-hot version of truth_array with the shape Nxnum_class
    """
    N = truth_array.shape[0]
    result = np.zeros((N, num_class))
    for i in range(N):
        result[i, truth_array[i]] = 1
    return torch.FloatTensor(result)


if __name__ == '__main__':

    start = time.time()

    training_data = torch.load("data/train_images.pt")
    training_ages = torch.load("data/train_ages.pt")
    training_genders = torch.load("data/train_genders.pt")
    # pre-process these data so they fit our model
    # normalize our data
    X_train = torch.from_numpy(training_data) / 255
    y_train_ages, y_train_genders = np2tensor(training_ages, training_genders)
    # print("The types of X_train, y_train_ages, y_train_genders are {}, {}, {}".format(type(X_train), type(y_train_ages), type(y_train_genders)))

    testing_data = torch.load("data/test_images.pt")
    testing_ages = torch.load("data/test_ages.pt")
    testing_genders = torch.load("data/test_genders.pt")
    # pre-process these data so they fit our model
    # normalize our data
    X_test = torch.from_numpy(testing_data) / 255
    y_test_ages, y_test_genders = np2tensor(testing_ages, testing_genders)


    # Here, I focused on the task of age estimation
    trainset = Data.TensorDataset(X_train[1000:6000], y_train_ages[1000:6000])
    train_loader = Data.DataLoader(trainset, batch_size=Batch_size, shuffle=True, num_workers=3)

    
    # below are the training and testing
    cnn = CNN(num_class=8)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.MSELoss()

    # print(y_train_ages.shape)
    # print(np.unique(y_train_ages[1000:5000], return_counts=True))
    for epoch in range(Epoches):
        print("Epoch {} begins".format(epoch))
        iter = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            iter += 1
            output = cnn(batch_x.float())
            # print(output.shape)
            # print(batch_y.shape)
            loss = loss_func(output, truth2label(batch_y, 8))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # display the validation accuracy
            if iter % 10 == 0:
                val_loss = float(loss.data)
                pred = torch.max(output, 1)[1].data.numpy()
                # print(pred)
                # print(batch_y.data.numpy())
                acc = ((pred == batch_y.data.numpy()).astype(int).sum()) / float(batch_y.size(0))
                print("Epoch: {} | train loss: {} | train accuracy: {}".format(epoch, round(val_loss, 4), round(acc, 2)))

    test_out = cnn(X_test[100:600].float())
    test_pred = torch.max(test_out, 1)[1].data.numpy()
    # print(test_pred)
    # print(type(test_pred))
    # print(y_test_ages[:100])
    # print(type(y_test_ages[:100]))
    test_acc = np.sum((test_pred == np.array(y_test_ages[100:600]))) / float(y_test_ages[100:600].size(0))
    print("The final test accuracy is {}".format(test_acc))


    end = time.time()
    print("The whole processing time is {}".format(end - start))