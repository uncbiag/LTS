import torch
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np
import glob
from Tiramisu_calibration_Dataset import *
from torch.utils.data import DataLoader
from calibration_models import *
from torch import nn, optim
import os
from tensorboardX import SummaryWriter
import time
import datetime
import os
import sys
import argparse
import random
sys.path.append(os.path.realpath(".."))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='index of used GPU')
parser.add_argument('--model-name', default='LTS', type=str, help='model name: IBTS, LTS, TS')
parser.add_argument('--epochs', default=200, type=int, help='max epochs')
parser.add_argument('--batch-size', default=4, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='inital learning rate')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--save-per-epoch', default=1, type=int, help='number of epochs to save model.')


if __name__ == "__main__":

    args = parser.parse_args()
    model_name = str(args.model_name)

    total_logits_list = glob.glob('/YOUR_PATH_TO_CamVid/prediction_results/val/*_logit.pt')
    total_logits_list.sort()

    ## training and validation split
    train_logits_list = total_logits_list[:90]
    val_logits_list   = total_logits_list[90:]

    nll_criterion = nn.CrossEntropyLoss()
    max_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    TIRAMISU_train = TIRAMISU_CALIBRATION(train_logits_list, 'val')
    TIRAMISU_train_dataloader = DataLoader(TIRAMISU_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    TIRAMISU_val = TIRAMISU_CALIBRATION(val_logits_list, 'val')
    TIRAMISU_val_dataloader = DataLoader(TIRAMISU_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if model_name == 'IBTS':
        experiment_name = model_name + '_CamVid' + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = IBTS_CamVid_With_Image()
    elif model_name == 'LTS':
        experiment_name = model_name + '_CamVid' + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = LTS_CamVid_With_Image()
    elif model_name == 'TS':
        experiment_name = model_name + '_CamVid' + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Temperature_Scaling()
    else:
        raise ValueError('Wrong Model Name!')


    calibration_model.weights_init()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        calibration_model.cuda(args.gpu)
    else:
        calibration_model.cuda()

    optimizer = optim.Adam(calibration_model.parameters(), lr=lr)

    print("Computing Loss")
    val_loss = 0
    for val_image, val_logits, val_labels, val_preds, val_boundary in TIRAMISU_val_dataloader:
        val_labels = val_labels.long().cuda(args.gpu)
        val_loss += nll_criterion(val_logits, val_labels).item()
    mean_val_loss = val_loss/len(TIRAMISU_val_dataloader)

    print('Before calibration - NLL: %.5f' % (mean_val_loss))

    calibration_model.train()
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    writer = SummaryWriter(os.path.join('./logs_CamVid', now_date, experiment_name + '_' + now_time))

    for epoch in range(max_epochs):
        for i, (train_image, train_logits, train_labels, train_preds, train_boundary) in enumerate(TIRAMISU_train_dataloader):
            global_step = epoch * len(TIRAMISU_train_dataloader) + (i + 1) * batch_size
            train_image, train_logits, train_labels = train_image.cuda(args.gpu), train_logits.cuda(args.gpu), train_labels.long().cuda(args.gpu)
            optimizer.zero_grad()
            logits_calibrate = calibration_model(train_logits, train_image, args)
            loss = nll_criterion(logits_calibrate, train_labels)
            loss.backward()
            optimizer.step()
            print("{} epoch, {} iter, training loss: {:.5f}".format(epoch, i + 1, loss.item()))
            writer.add_scalar('loss/training', loss.item(), global_step=global_step)

            ## save the current best model and checkpoint
            if i%10 == 9 and epoch % args.save_per_epoch == (args.save_per_epoch - 1):
                with torch.set_grad_enabled(False):
                    tmp_loss = 0
                    for val_image, val_logits, val_labels, val_preds, val_boundary in TIRAMISU_val_dataloader:
                        val_image, val_logits, val_labels = val_image.cuda(args.gpu), val_logits.cuda(args.gpu), val_labels.long().cuda(args.gpu)
                        logits_cali = calibration_model(val_logits, val_image)
                        tmp_loss += nll_criterion(logits_cali, val_labels).item()
                    mean_tmp_loss = tmp_loss/len(TIRAMISU_val_dataloader)
                    print("{} epoch, {} iter, training loss: {:.5f}, val loss: {:.5f}".format(epoch, i+1, loss.item(), mean_tmp_loss))
                    writer.add_scalar('loss/validation', mean_tmp_loss, global_step=global_step)

                    if mean_tmp_loss < mean_val_loss:
                        mean_val_loss = mean_tmp_loss
                        print('%d epoch, current lowest - NLL: %.5f' % (epoch, mean_val_loss))
                        writer.add_scalar('validation/lowest_loss', mean_val_loss, global_step=global_step)
                        torch.save(calibration_model.state_dict(), './calibration_Tiramisu/' + experiment_name + '_params.pth.tar')
                        best_state = {'epoch': epoch,
                                      'state_dict': calibration_model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'best_score': mean_val_loss,
                                      'global_step': global_step
                                     }
                        torch.save(best_state, './calibration_Tiramisu/' + experiment_name + '_model_best.pth.tar')

                    current_state = {'epoch': epoch,
                                     'state_dict': calibration_model.state_dict(),
                                     'optimizer': optimizer.state_dict(),
                                     'best_score': mean_tmp_loss,
                                     'global_step': global_step
                                    }
                    torch.save(current_state, './calibration_Tiramisu/' + experiment_name + '_checkpoint.pth.tar')





