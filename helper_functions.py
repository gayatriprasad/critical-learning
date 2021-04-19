import os
import csv
import torch
import pickle
import copy
from copy import deepcopy
import random
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import no_grad
from torch.autograd import Variable


def diag_fisher(model, dataset, dataloader, criterion, optimizer, cuda, device_id):

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    precision_matrices = {}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = Variable(p.data)
    model.eval()
    # for input in dataset:
    for images, labels in dataloader:
        if cuda and device_id in [0, 1, 2, 3]:
            images = Variable(images).cuda(device_id)
            labels = Variable(labels).cuda(device_id)
        # setting all gradients to zero
        optimizer.zero_grad()
        # run model on input
        output = model(images)
        # calculate the loss
        loss = criterion(output, labels)
        # update gradients
        loss.backward()
        # update loss
        optimizer.step()

        for n, p in model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(dataset)

    precision_matrices = {n: p for n, p in precision_matrices.items()}

    print('precision_matrices', precision_matrices)
    # return precision_matrices


def unpickle(file):
    """Load byte data from file"""
    with open(file, 'rb') as f:  # rb for reading, writing in binary form
        data = pickle.load(f, encoding='latin-1')  # latin-1 encoding for Numpy arrays
        return data


def export_history(value, folder, file_name):
    """ Export data to csv format
    Args:
        value (list): values of corresponding columns
        folder (list): folder path
        file_name: file name with path
    """
    # if folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    file = open(folder + file_name, 'a', newline='')
    writer = csv.writer(file)
    writer.writerows(value)
    # close file when it is done with writing
    file.close()


def save_model(model, model_save_path, model_save_file_name):
    """ Save model to given path
    Args:
        model: model to be saved
        model_save_path: path that the model would be saved
        model_save_file_name: name that the model would be saved with
        epoch: the epoch the model finished training
    """
    # This is to avoid shuffling the model from cpu to gpu
    model_copy = copy.deepcopy(model)
    model_copy.cpu()
    model_copy.eval()
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model_copy.state_dict(), model_save_path + model_save_file_name)


def load_model(model, model_load_path, model_save_file_name):
    """ Load model from given path
    Args:
        model_load_path: path that the model would be loaded from
        model_save_file_name : name of the file to be loaded
    """
    model.load_state_dict(torch.load(model_load_path + model_save_file_name))
    model.cpu()
    model.eval()
    return model


def crop_and_pad(img_as_np, crop_size):
    """
  pad an image and crop it to original size
   Args:
        images: numpy array of images
        crop_size(int): size of cropped image(32)
    Return:
        padded and cropped image: numpy array of padded and cropped image
    """
    # print('img_as_np_before_pad', img_as_np.shape[0], img_as_np.shape[1], img_as_np.shape[2])
    pad_size = 4
    img_as_np = np.pad(img_as_np, ((0, 0), (pad_size, pad_size),
                                   (pad_size, pad_size)), mode="symmetric")
    # print('img_as_np_after_pad', img_as_np.shape[0], img_as_np.shape[1], img_as_np.shape[2])
    y_loc, x_loc = random.randint(0, pad_size), random.randint(0, pad_size)
    cropped_img = img_as_np[::, y_loc:y_loc + crop_size, x_loc:x_loc + crop_size]
    # print('cropped_img', cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2])
    return cropped_img


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added and with maximum-255 and minimum-0
    """
    # image = image.astype("int16")
    image = image + value
    image[image > 255] = 255
    image[image < 0] = 0
    return image


def horizontal_flip(image):
    """
    Args:
        image : numpy array of image
    Return :
        image : numpy array of flipped image
    """
    # horizontal
    flipped_image = np.flip(image, 1)
    return flipped_image


def vertical_flip(image):
    """
    Args:
        image : numpy array of image
    Return :
        image : numpy array of flipped image
    """
    # horizontal
    flipped_image = np.flip(image, 0)
    return flipped_image


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calculate_accuracy_from_csv(folder_path, upper_limit, tr_ts='tr'):
    """ Calculate the accuracy of the model from csv files
    Args:
        folder_path: path that the csv are saved in
        upper_limit: last epoch number for csv files saved
        tr_ts: train or test accuracy flag
    """
    csv_list = []
    # Read files in an ordered way, if they exist
    for i in range(upper_limit):
        file_name = folder_path + '/' + tr_ts + '_logit_epoch_' + str(i) + '.csv'
        if os.path.isfile(file_name):
            csv_list.append(file_name)

    acc_list = []
    for single_csv in csv_list:
        # Read csv
        csv_as_np = pd.read_csv(single_csv).values
        # Get true class
        true_class = csv_as_np[:, 0]
        # Get logic preds
        logit_preds = csv_as_np[:, 1:]
        logit_max = np.argmax(logit_preds, axis=1)
        true_class = csv_as_np[:, 0]
        # Compare preds
        acc = true_class == logit_max
        tot = true_class == true_class
        acc = np.sum(acc) / np.sum(tot)
        # Add to list
        acc_list.append(acc)
    # Convert to np array
    np_acc = np.asarray(acc_list)
    # sanity check
    """
    print('np_acc', np_acc)
    print('len_acc', len(np_acc))
    """
    # Save np array
    np.save(folder_path + '/' + tr_ts + '_' + 'accuracy_array.npy', np_acc)


def load_accuracy_numpy_array_per_model(work_dir, model_ver, SEED, w_init, reset_flag, upper_limit, tr_ts='ts'):
    """ Reads folder list Loads the accuracy per model from directory
    Args:
        work_dir: path that the models are saved in
        model_ver: model version
        tr_ts: train or test accuracy flag
        SEED: version
        w_init: Weight initialization
        reset_flag: Resetting parameters
        upper_limit:last epoch number for model
    """
    folder_list = []
    # Read folders in an ordered way, if they exist
    for i in range(0, 201, 20):
        folder_name = work_dir + '/model(' + model_ver + ')_init(' + w_init + ')_v(' + str(
            SEED) + ')_defrem(' + str(i) + ')' + '_reset_param(' + str(reset_flag) + ')/'
        if os.path.isdir(folder_name):
            folder_list.append(folder_name)

    maximum_acc_list = []  # maximum accuracy per model list
    final_acc_list = []  # final accuracy per model list

    for single_folder in folder_list:
        # change directories
        os.chdir(single_folder)
        # load the numpy array
        acc_as_np = np.load('ts_accuracy_array.npy')
        # find the maximum accuracy
        maximum_acc_list.append(np.max(acc_as_np))
        # the final accuracy (at last epoch)
        final_acc_list.append(acc_as_np[-1])

    # sanity check
    """
    print('max_acc_list:', max_acc_list)
    print()
    # print('len_max_acc_list:', len(max_acc_list))
    print()
    print('final_acc_list:', final_acc_list)
    """
    return maximum_acc_list, final_acc_list


def plot_accuracy(maximum_acc_list, final_acc_list):

    deficit_removal_epoch = np.arange(0, 201, 20)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.xticks(deficit_removal_epoch)
    plt.ylim(0.5, 0.9)
    plt.plot(deficit_removal_epoch, maximum_acc_list, '-r', label='maximum_acc_list', marker='.')
    plt.plot(deficit_removal_epoch, final_acc_list, '-g', label='final_acc_list', marker='.')
    plt.xlabel('Deficit Removal Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Dependency on Model Accuracy on Deficit Removal')
    plt.show()
