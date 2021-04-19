# import libraries
import torch
from torch import from_numpy
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from helper_functions import unpickle, horizontal_flip, crop_and_pad, change_brightness
import numpy as np
import random
import copy
from PIL import Image


class CifarDataset(Dataset):
    """ Returns data, labels
        The shape of data is 3 x 32 x 32
        The labels are of type integer
    """

    def __init__(self, data_dir, train_test_flag, blur_flag):
        """
        Args:
            data_dir (string): path to data file
            train_test_flag: flag to indicate if it is train data or test data
            blur_flag: flag to indicate if the data will be blurred or not
        """
        # reading the file and converting it to numpy array
        self.data = None
        self.labels = []
        self.blur_flag = blur_flag
        self.train_test_flag = train_test_flag

        # Get train data
        if self.train_test_flag == 'train':
            for i in range(1, 6):
                data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
                if i == 1:
                    self.data = data_dic['data']
                else:
                    self.data = np.vstack((self.data, data_dic['data']))
                self.labels += data_dic['labels']
        # Get test data
        elif self.train_test_flag == 'test':
            data_dic = unpickle(data_dir + "/test_batch")
            self.data = data_dic['data']
            self.labels = data_dic['labels']

        # Data processing for both train and test
        self.data = self.data.reshape((len(self.data), 3, 32, 32))
        self.data = np.rollaxis(self.data, 1, 4)
        # length of the test data
        self.data_len = len(self.data)
        print('Dataset initialized with', self.data_len, 'samples.')

    def __getitem__(self, index):
        """Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        # Get image
        single_image = self.data[index]
        # Open Image
        single_image = Image.fromarray(single_image)
        if self.blur_flag:
            # Downsampling the image
            single_image = single_image.resize((8, 8))
            # Upsampling the image using BILINEAR interpolation
            single_image = single_image.resize((32, 32), Image.BILINEAR)
        """
        #  ---- sanity check begins ----
        single_image.show()
        #  ---- sanity check ends ----
        """
        # check if image channels are RGB
        # print('channels of image', single_image.mode)
        # converting to a numpy array
        single_image = np.asarray(single_image)
        # apply tranpose to change shape to C-H-W
        single_image = single_image.transpose(2, 0, 1)
        # horizontal flip with probability = 0.5
        if random.random() >= 0.8 and self.train_test_flag == 'train':
            single_image = horizontal_flip(single_image)
        # random crop with probability = 0.5
        if random.random() >= 0.8 and self.train_test_flag == 'train':
            single_image = crop_and_pad(single_image, 32)
        # Brightness
        if random.random() >= 0.5 and self.train_test_flag == 'train':
            pixel_brightness = random.randint(-20, 20)
            single_image = change_brightness(single_image, pixel_brightness)
        # normalizing images with standard values for Cifar10
        mean = [0.4465, 0.4914, 0.4822]
        std_dev = [0.2010, 0.2023, 0.1994]
        single_img = (copy.copy(single_image)).astype(np.float32)
        single_img /= 255
        for c in range(3):
            single_img[c] /= std_dev[c]
            single_img[c] -= mean[c]
        # apply transforms- convert to tensor
        single_img = np.ascontiguousarray(single_img)
        single_image_tensor = from_numpy(single_img).float()
        # Get label(class) of the image
        single_image_label = self.labels[index]
        # print(type(single_image_label))

        return (single_image_tensor, single_image_label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    # pass

    data_dir = '/home/gp/Documents/projects/cifar/data/cifar-10-batches-py'

    train_dt = CifarDataset(data_dir, 'train', 'a')
    train_dt[0]
    """
    train_dt = CifarDataset(data_dir, 'train', False)
    train_dt[0]
    train_dt = CifarDataset(data_dir, 'train', 'abc')
    train_dt[0]
    test_dt = CifarDataset(data_dir, 'test', False)
    test_dt[0]
    """
