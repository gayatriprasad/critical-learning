import torch
import numpy as np
from torch import no_grad
from torch.autograd import Variable
from helper_functions import export_history

TRAIN_HEADER = ['true_class', 'plane_(0)', 'car_(1)', 'bird_(2)', 'cat_(3)',
                'deer_(4)', 'dog_(5)', 'frog_(6)', 'horse_(7)', 'ship_(8)', 'truck_(9)']


def train_one_epoch(model, tr_loader, criterion, optimizer, cuda=False, device_id=-1):
    model.train()
    for images, labels in tr_loader:
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


def make_prediction(model, data_loader,  predictions_save_dir, file_name, cuda=False, device_id=-1):
    export_history([TRAIN_HEADER], predictions_save_dir, file_name)
    model.eval()
    acc = 0
    for j, (data) in enumerate(data_loader):
        images, labels = data
        if cuda and device_id in [0, 1, 2, 3]:
            images = Variable(images).cuda(device_id)
            labels = Variable(labels).cuda(device_id)
        with no_grad():  # Turning off gradients
            output = model(images)
        _, preds = torch.max(output, 1)
        preds = preds.cpu()
        correct_pred = preds == labels.cpu()
        acc = acc + torch.sum(correct_pred)
        # get the true class as well as the predictions and export history to csv
        true_class_with_logits = list(np.c_[labels.cpu(), output.cpu().detach().numpy()])
        print('true_class_with_logits', true_class_with_logits)

        export_history(true_class_with_logits, predictions_save_dir, file_name)
    acc = acc.item() / len(data_loader.dataset)
    print(file_name, 'ACC:', str(acc))
