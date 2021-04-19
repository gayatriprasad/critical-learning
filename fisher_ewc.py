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

SEED = 3
torch.manual_seed(SEED)

if __name__ == '__main__':

    data_dir = '/home/saiperi/critical-learning/data/cifar-10-batches-py'
    # Dataset
    tr_dataset = CifarDataset(data_dir, 'train', True)
    tr_dataset_len = len(tr_dataset)
    # Dataloader
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=128, num_workers=16, shuffle=True)
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

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    precision_matrices = {}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = Variable(p.data)

    for images, labels in tr_loader:
        if cuda and device_id in [0, 1, 2, 3]:
            images = Variable(images).cuda(device_id)
            labels = Variable(labels).cuda(device_id)
            print('labels :', labels)
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
        # Generate gradients
        for n, p in model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2

    precision_matrices = {n: (p.cpu().numpy().sum()) /
                          tr_dataset_len for n, p in precision_matrices.items()}
    # activations, gradients = FI.generate_activations_gradients(images, labels)
    #print('gradients shape :', len(gradients))
    # print('activations shape :', len(activations))
    # print('gradients  :', gradients)
    print('precision_matrices  :', precision_matrices)
