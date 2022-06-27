import os
import math
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


def data_finder(file_path):
    """
    Return a tuple of 4 lists, each containing the directory name, file name, age and gender of each image.
    """
    dir_lst, file_lst, age_lst, gender_lst = [], [], [], []
    with open(file_path, "r") as file:
        x = file.readlines()
        raw_info = x[1:]
        for line in raw_info:
            line_info = line.split()
            if line_info[5] == "m" or line_info[5] == "f":      # only take the part where the gender is clear
                # extract the information of dir + file name
                dir_name, file_name, gender = line_info[0], line_info[1], line_info[5]
                l_age, r_age = (line_info[3] + line_info[4])[1:-1].split(",")
                l_age, r_age = int(l_age), int(r_age)
                # ================ data correction ================
                # The age labels in the original dataset is not corresponding to the reported categories
                # The following corrections are based on the possible typos in the datasets
                if l_age == 8:
                    r_age = 13
                if r_age == 32:
                    l_age = 25
                if l_age == 38:
                    r_age = 43
                # ================ =============== ================
                age = str((l_age, r_age))
                dir_lst.append(dir_name)
                file_lst.append(line_info[2] + "." + file_name)         # line_info[2] is an id in the file name
                age_lst.append(age)
                gender_lst.append(gender)
    return dir_lst, file_lst, age_lst, gender_lst


def data_loader(dir_lst, file_lst, age_lst, gender_lst):
    """
    Return an array of shape (N, C, H, W) as the image data, an array of shape (N,) as ages and another as genders
    """
    X, Y_age, Y_gender = [], [], []
    for i in range(len(file_lst)):
        # extract the image data
        dir, file = dir_lst[i], file_lst[i]
        fname = os.path.join("faces", dir, "coarse_tilt_aligned_face." + file)
        curr_image = Image.open(fname)
        crop_image = curr_image.resize((227, 227), Image.ANTIALIAS)         # crop the image to fit our model  
        crop_array = np.array(crop_image)                       # convert the image to ndarray and normalize
        final_array = np.transpose(crop_array, (2, 0, 1))       # change the image to be C, H, W
        X.append(final_array)
        # extract the labels: age_range, gender
        age, gender = age_lst[i], gender_lst[i]
        Y_age.append(age)
        Y_gender.append(gender)

    images = np.array(X)      # now the shape is N, C, H, W
    ages = np.array(Y_age)    # now the shape is N, 
    genders = np.array(Y_gender)        # now the shape is N,

    return images, ages, genders


if __name__ == '__main__':

    start = time.time()

    all_images, all_ages, all_genders = [], [], []
    for i in range(5):
        fname = "fold_" + str(i) + "_data.txt"
        dir, file, age, gender = data_finder(fname)
        images, ages, genders = data_loader(dir, file, age, gender)
        all_images.append(images)
        all_ages.append(ages)
        all_genders.append(genders)

    # prepare as the training data
    training_data = np.concatenate(all_images[:-1], axis=0)
    training_ages = np.concatenate(all_ages[:-1], axis=0)
    training_genders = np.concatenate(all_genders[:-1], axis=0)
    print("The training data shape is {}".format(training_data.shape))
    print("The training ages shape is {}".format(training_ages.shape))
    print("The training genders shape is {}".format(training_genders.shape))

    # prepare as the testing data
    testing_data = np.array(all_images[-1])
    testing_ages = np.array(all_ages[-1])
    testing_genders = np.array(all_genders[-1])
    print("The testing data shape os {}".format(testing_data.shape))
    print("The testing ages shape is {}".format(testing_ages.shape))
    print("The testing genders shape is {}".format(testing_genders.shape))

    print("The lables of ages are {}".format(np.unique(np.concatenate([training_ages, testing_ages], axis=0))))
    print("The lables of genders are {}".format(np.unique(np.concatenate([training_genders, testing_genders], axis=0))))

    # save the data for further usage
    torch.save(training_data, "data/train_images.pt")
    torch.save(training_ages, "data/train_ages.pt")
    torch.save(training_genders, "data/train_genders.pt")

    torch.save(testing_data, "data/test_images.pt")
    torch.save(testing_ages, "data/test_ages.pt")
    torch.save(testing_genders, "data/test_genders.pt")

    end = time.time()
    print("The whole processing time is {}".format(end - start))