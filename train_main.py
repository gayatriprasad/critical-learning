# torch import
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# other libraries
import argparse

# Import from local files
from cifar_dataset import CifarDataset
from model import OrgCifarModel, resnet20, xavier_initialization_weights
from modules import train_one_epoch, make_prediction
from helper_functions import save_model, calculate_accuracy_from_csv, load_model, plot_accuracy, diag_fisher


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

# BEST SGD
"""
parser.add_argument('--blur_stop_epoch', default=0, type=int, metavar='N',
                    help='total blur epochs')
parser.add_argument('--DEVICE_ID', default=2, type=int,
                    metavar='DEVICE_ID', help='Device ID for GPU')
parser.add_argument('--model_load_flag', '--model_load_flag', default=False,
                    metavar='model_load_flag', help='model load flag for after removal')
parser.add_argument('--model_load_epoch', default=0, type=int,
                    metavar='model_load_epoch', help='model epoch to be loaded')
parser.add_argument('--step_size', default=80, type=int, metavar='step_size',
                    help='change learning every step_size epochs')
parser.add_argument('--gamma', default=0.1, type=float, metavar='gamma',
                    help='change learning rate to lr*gamma every step_size epochs')
parser.add_argument('--learning_rate', default=0.1, type=float, metavar='learning_rate',
                    help='learning_rates')
parser.add_argument('--weight_decay', default=0.0005, type=float, metavar='weight_decay',
                    help='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--reset_parameters_flag', default=True,
                    metavar='reset_parameters_flag', help='flag to reset parameters')

parser.add_argument('--init', default='def', metavar='initialization', help='initialization')


# Paper Adam


parser.add_argument('--blur_stop_epoch', default=0, type=int, metavar='N',
                    help='total blur epochs')
parser.add_argument('--DEVICE_ID', default=2, type=int,
                    metavar='DEVICE_ID', help='Device ID for GPU')
parser.add_argument('--model_load_flag', '--model_load_flag', default=False,
                    metavar='model_load_flag', help='model load flag for after removal')
parser.add_argument('--model_load_epoch', default=0, type=int,
                    metavar='model_load_epoch', help='model epoch to be loaded')
parser.add_argument('--learning_rate', default=0.001, type=float, metavar='learning_rate',
                    help='learning_rates')
parser.add_argument('--weight_decay', default=0.0001, type=float, metavar='weight_decay',
                    help='weight_decay')
parser.add_argument('--reset_parameters_flag', default=True,
                    metavar='reset_parameters_flag', help='flag to reset parameters')
parser.add_argument('--init', default='def', metavar='initialization', help='initialization')
"""
# Paper SGD
parser.add_argument('--blur_stop_epoch', default=0, type=int, metavar='N',
                    help='total blur epochs')
parser.add_argument('--DEVICE_ID', default=0, type=int,
                    metavar='DEVICE_ID', help='Device ID for GPU')
parser.add_argument('--model_load_flag', '--model_load_flag', default=False,
                    metavar='model_load_flag', help='model load flag for after removal')
parser.add_argument('--model_load_epoch', default=0, type=int,
                    metavar='model_load_epoch', help='model epoch to be loaded')
parser.add_argument('--step_size', default=1, type=int, metavar='step_size',
                    help='change learning every step_size epochs')
parser.add_argument('--gamma', default=0.97, type=float, metavar='gamma',
                    help='change learning rate to lr*gamma every step_size epochs')
parser.add_argument('--learning_rate', default=0.05, type=float, metavar='learning_rate',
                    help='learning_rates')
parser.add_argument('--weight_decay', default=0.001, type=float, metavar='weight_decay',
                    help='weight_decay')
parser.add_argument('--reset_parameters_flag', default=True,
                    metavar='reset_parameters_flag', help='flag to reset parameters')
parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')
parser.add_argument('--init', default='def', metavar='initialization', help='initialization')


SEED = 3
torch.manual_seed(SEED)


if __name__ == "__main__":

    global args
    args = parser.parse_args()

    # Dataset
    tr_dataset = CifarDataset(
        '/home/saiperi/critical-learning/data/cifar-10-batches-py', 'train', True)
    ts_dataset = CifarDataset(
        '/home/saiperi/critical-learning/data/cifar-10-batches-py', 'test', False)

    # Dataloader
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=128, num_workers=16, shuffle=True)
    print('len_train_loader: ', len(tr_loader))
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=128, num_workers=16, shuffle=False)

    # Model
    # Paper model
    model = OrgCifarModel()
    # Resnet-18
    # model = resnet20(init='he')

    # Xavier init or Default init
    w_init = args.init
    print('w_init', w_init)  #
    if w_init == 'xav':
        model.apply(xavier_initialization_weights)

        # Params that change
    blur_stop_epoch = args.blur_stop_epoch
    print('blur_stop_epoch:', blur_stop_epoch)
    DEVICE_ID = args.DEVICE_ID
    print('DEVICE_ID:', DEVICE_ID)
    reset_parameters_flag = args.reset_parameters_flag
    print('reset_parameters_flag:', reset_parameters_flag)
    # step_size = args.step_size
    # print('step_size:', step_size)
    # gamma = args.gamma
    # print('gamma:', gamma)
    learning_rate = args.learning_rate
    print('learning_rate:', learning_rate)
    weight_decay = args.weight_decay
    print('weight_decay:', weight_decay)

    # Static params
    cuda = True
    make_pred_epoch = 1
    total_epoch = args.blur_stop_epoch + 2
    print('total_epoch:', total_epoch)
    model_ver = 'resnet' if type(model).__name__ == 'ResNet' else 'paper'

    # Save directories;
    work_dir = '/home/saiperi/critical-learning/fim_results'
    save_dir = '/home/saiperi/critical-learning/fim_results(' + model_ver + ')_init(' + w_init + \
        ')_v(' + str(SEED) + ')_defrem(' + str(blur_stop_epoch) + \
        ')_reset_param(' + str(args.reset_parameters_flag) + ')/'
    model_save_dir = save_dir + 'models/'

    # Load model
    if args.model_load_flag:
        model_load_path = model_save_dir = save_dir + 'models/'
        model_file_name = 'model(' + model_ver + ')_init(' + w_init + ')_v(' + str(SEED) + ')_defrem(' + str(blur_stop_epoch) + ')_'\
            + 'epoch_' + str(args.model_load_epoch) + '.pt'
        model = load_model(model, model_load_path, model_save_file_name)

    # Use single gpu
    if cuda:
        model.cuda(DEVICE_ID)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)  # step_size = every nth epoch

    for epoch in range(total_epoch + 1):
        # Remove deficit when the time is right
        if epoch >= blur_stop_epoch:
            tr_loader.dataset.blur_flag = False
        else:
            tr_loader.dataset.blur_flag = True

        # Return the parameters to original after blur epochs
        if epoch == blur_stop_epoch and reset_parameters_flag == True:
            print('Reverting to original parameters')
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
            # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.step_size,
                               gamma=args.gamma)  # step_size = every nth epoch

        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        # print('Epoch:', epoch, 'LR:', args.learning_rate)
        train_one_epoch(model, tr_loader, criterion, optimizer, cuda=cuda, device_id=DEVICE_ID)
        scheduler.step()  # LR annealing after every epoch

        if epoch % make_pred_epoch == 0:
            print('Test starts')
            # No deficit on train set for make_prediction
            tr_loader.dataset.blur_flag = False

            # Model and csv file names
            train_file_name = 'tr_logit_epoch_' + str(epoch) + '.csv'
            test_file_name = 'ts_logit_epoch_' + str(epoch) + '.csv'
            model_file_name = 'model(' + model_ver + ')_init(' + w_init + ')_v(' + str(SEED) + ')_defrem(' + str(blur_stop_epoch) + ')_'\
                + 'epoch_' + str(epoch) + '.pt'

            # Train prediction
            make_prediction(model, tr_loader, save_dir, train_file_name,
                            cuda=cuda, device_id=DEVICE_ID)
            # Test prediction
            make_prediction(model, ts_loader, save_dir, test_file_name,
                            cuda=cuda, device_id=DEVICE_ID)
            # Save model
            # save_model(model, model_save_dir, model_file_name)
            print('Test ends')

    calculate_accuracy_from_csv(save_dir, tr_ts='tr', upper_limit=total_epoch)
    calculate_accuracy_from_csv(save_dir, tr_ts='ts', upper_limit=total_epoch)

    # fisher information
    diag_fisher(model, tr_dataset, tr_loader, criterion, optimizer, cuda=cuda, device_id=DEVICE_ID)

print('Fin.')
