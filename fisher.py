import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from model import OrgCifarModel
from cifar_dataset import CifarDataset
from torch.nn import Linear, Conv2d
from torch.utils.data import DataLoader
import copy
from helper_functions import export_history
from copy import deepcopy
from torch.autograd import Variable
import matplotlib.pylab as plt
import numpy as np
import os
from keras.utils import to_categorical


SEED = 3
torch.manual_seed(SEED)


class FisherInformation():
    """
    Produces trace of the Fisher Information matrix at various layers
    """

    def __init__(self, model):
        self.model = model
        self.gradients = []
        self.activations = []
        # put model in eval mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def forward_hook_function(module, activ_in, activ_out):
            """
            Store results of forward pass
            """
            self.activations = activ_out

        def backward_hook_function(module, grad_in, grad_out):
            """
            Store results of backward pass
            """
            self.gradients = grad_in[1]

        # Loop through layers and register forward and backward hooks to all the convolution layers
        for pos, module in model.features._modules.items():
            if isinstance(module, Conv2d):
                module.register_backward_hook(backward_hook_function)
                module.register_forward_hook(forward_hook_function)
                # print('Hooks successfully registered')

    def generate_activations_gradients(self, input, target_class):
        # Forward
        model_output = self.model(input)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output_as_np = to_categorical(target_class.cpu().numpy())
        one_hot_output_as_tensor = torch.from_numpy(one_hot_output_as_np).to(device)
        # Backward pass
        model_output.backward(gradient=one_hot_output_as_tensor)
        # convert activations and gradients into a list of numpy arrays
        gradients_as_np = [x.cpu().data.numpy() for x in self.gradients]
        activations_as_np = [x.cpu().data.numpy() for x in self.activations]
        return (activations_as_np, gradients_as_np)


def open_read_numpy_files(folder_path):
    """
    opens each activations/gradients numpy file and calculates the
    product of the activations/gradients per layer
    """
    # change directories
    os.chdir(folder_path)

    grad_files_list = []
    activ_files_list = []
    file_name = None
    # Read activation files in an ordered way, if they exist
    for epoch in range(0, 360):  # read till total number of epochs
        for iter in range(0, 391):  # 50000/batch_size(128) = 390
            file_name = folder_path + '/epoch_' + \
                str(epoch) + '_iteration_' + str(iter) + '_activations_array.npy'
            activ_files_list.append(file_name)
            # print('Activ')
    # Read activation files in an ordered way, if they exist
    for epoch in range(0, 360):  # read till total number of epochs
        for iter in range(0, 391):  # 50000/batch_size(128) = 390
            file_name = folder_path + '/epoch_' + \
                str(epoch) + '_iteration_' + str(iter) + '_gradients_array.npy'
            grad_files_list.append(file_name)
            # print('Grad')
    #print('activ_files_list', activ_files_list)
    #print('grad_files_list', grad_files_list)

    return grad_files_list, activ_files_list


def calculate_activ_grad_per_layer(grad_files_list, activ_files_list):

    activ_per_layer = []
    grad_per_layer = []
    for single_np_file in grad_files_list:
        # change directories
        os.chdir(folder_path)
        np_file = np.load(single_np_file)
        print('np_file_shape : ', np_file.shape)
        


if __name__ == '__main__':

    data_dir = '/home/saiperi/critical-learning/data/cifar-10-batches-py'
    folder_path = '/home/saiperi/critical-learning/fim_results'
    """
    # if folder does not exists make folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Dataset
    tr_dataset = CifarDataset(data_dir, 'train', True)
    tr_dataset_len = len(tr_dataset)
    # Dataloader
    batch_size = 128
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
    # get model
    model = OrgCifarModel()
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    cuda = True
    device_id = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Fisher Information
    FI = FisherInformation(model)

    for epoch in range(0, 2):
        for idx, data in enumerate(tr_loader):
            images = data[0]
            labels = data[1]
            if cuda and device_id in [0, 1, 2, 3]:
                images = Variable(images).cuda(device_id)
                labels = Variable(labels).cuda(device_id)

            # generate_activations and gradients
            activations, gradients = FI.generate_activations_gradients(images, labels)
            # Save activations np array
            np.save(folder_path + '/' + 'epoch' + '_' + str(epoch) + '_iteration_' +
                    str(idx) + '_' + 'activations_array.npy', activations)
            # Save gradients np array
            np.save(folder_path + '/' + 'epoch' + '_' + str(epoch) + '_iteration_' +
                    str(idx) + '_' + 'gradients_array.npy', gradients)
    """
    # calcuulate Fisher Diagonals
    grad_files_list, activ_files_list = open_read_numpy_files(folder_path)
    calculate_activ_grad_per_layer(grad_files_list, activ_files_list)
