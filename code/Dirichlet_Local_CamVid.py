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
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("../.."))
sys.path.insert(1, '../dirichlet_python')
sys.path.insert(1, '../experiments_neurips')
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from probability_measure_CamVid import Calculate_ECE, Calculate_MCE, Calculate_SCE, Calculate_ACE
import pickle
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from dirichletcal.calib.fixeddirichlet import FixedDiagonalDirichletCalibrator
from calib.models.dirichlet_keras import Dirichlet_NN
from keras.models import load_model
import random

total_logits_list = glob.glob('/YOUR_PATH_TO_CamVid/results/val/*_logit.pt')
total_logits_list.sort()
total_logits_test_list = glob.glob('/YOUR_PATH_TO_CamVid/results/test/*_logit.pt')
total_logits_test_list.sort()

train_logits_list = total_logits_list[:90]
val_logits_list   = total_logits_list[90:]

torch.cuda.manual_seed(0)

TIRAMISU_train = TIRAMISU_CALIBRATION(train_logits_list, 'val')
TIRAMISU_train_dataloader = DataLoader(TIRAMISU_train, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)
TIRAMISU_val = TIRAMISU_CALIBRATION(val_logits_list, 'val')
TIRAMISU_val_dataloader = DataLoader(TIRAMISU_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
TIRAMISU_test = TIRAMISU_CALIBRATION(total_logits_test_list, 'test')
TIRAMISU_test_dataloader = DataLoader(TIRAMISU_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

print("merge individual cases!")
all_probs = None
all_labels = None
for i, (val_image, val_logits, val_labels, val_preds, val_boundary) in enumerate(TIRAMISU_val_dataloader):
    test_probs = torch.softmax(val_logits, dim=1).detach().squeeze().cpu().numpy()
    val_label_array = val_labels.detach().squeeze().cpu().numpy()
    prob_img_array_select = np.transpose(test_probs.reshape((12, -1)))
    val_label_array_select = val_label_array.reshape(-1)
    # val_label_array_select_onehot = np.eye(12)[val_label_array_select]
    if all_probs is None:
        all_probs = prob_img_array_select
        # all_labels = val_label_array_select_onehot
        all_labels = val_label_array_select
    else:
        all_probs = np.concatenate((all_probs, prob_img_array_select), axis=0)
        # all_labels = np.concatenate((all_labels, val_label_array_select_onehot), axis=0)
        all_labels = np.concatenate((all_labels, val_label_array_select))


print('start training!')
# l2_odir = 1e-2
# dirichlet_calibration = FullDirichletCalibrator(reg_lambda=l2_odir, reg_mu=l2_odir, reg_norm=False)
# dirichlet_calibration = FixedDiagonalDirichletCalibrator()
dirichlet_calibration = Dirichlet_NN(l2=0.001, classes=12, comp=True, max_epochs=20, patience=3, lr=0.0001)
dirichlet_calibration.fit(all_probs, all_labels)

# dirichlet_calibration.save('./calibration/dirichlet_prob.h5')
# dirichlet_calibration.load_model('./calibration/dirichlet_prob.h5')

res_list_Local_ECE = []
res_list_Local_MCE = []
res_list_Local_SCE = []
res_list_Local_ACE = []

## add local patch center
random.seed(10)
patch_len = 36

for ind, (test_image, test_logits, test_labels, test_preds, test_boundary) in enumerate(TIRAMISU_test_dataloader):
    print(ind)

    image_shape = test_logits.squeeze().shape

    test_probs = np.transpose(torch.softmax(test_logits, dim=1).detach().squeeze().cpu().numpy().reshape((12, -1)))
    dirichlet_correction = np.transpose(dirichlet_calibration.predict(test_probs)).reshape(image_shape)

    gt_img_array   = test_labels.squeeze().cpu().numpy()
    boundary_img_array = test_boundary.squeeze().cpu().numpy()

    prob_img_array = np.max(dirichlet_correction, axis=0)/np.sum(dirichlet_correction, axis=0)
    pred_img_array = np.argmax(dirichlet_correction, axis=0)

    for lo_ind in range(10):
        random_center = (
        random.randint(84, pred_img_array.shape[0] - 84), random.randint(84, pred_img_array.shape[1] - 84))

        patch_prob_img_array = prob_img_array[random_center[0] - patch_len:random_center[0] + patch_len, random_center[1] - patch_len:random_center[1] + patch_len]
        patch_pred_img_array = pred_img_array[random_center[0] - patch_len:random_center[0] + patch_len, random_center[1] - patch_len:random_center[1] + patch_len]
        patch_gt_img_array = gt_img_array[random_center[0] - patch_len:random_center[0] + patch_len, random_center[1] - patch_len:random_center[1] + patch_len]

        res_list_Local_ECE.append(Calculate_ECE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
        res_list_Local_MCE.append(Calculate_MCE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
        res_list_Local_SCE.append(Calculate_SCE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
        res_list_Local_ACE.append(Calculate_ACE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))

with open("./CamVid_result/"+"Dirichlet_CamVid_ICCV_Local_36_ECE.txt", "wb") as fp_ECE:
    pickle.dump(res_list_Local_ECE, fp_ECE, protocol=2)
with open("./CamVid_result/"+"Dirichlet_CamVid_ICCV_Local_36_MCE.txt", "wb") as fp_MCE:
    pickle.dump(res_list_Local_MCE, fp_MCE, protocol=2)
with open("./CamVid_result/"+"Dirichlet_CamVid_ICCV_Local_36_SCE.txt", "wb") as fp_SCE:
    pickle.dump(res_list_Local_SCE, fp_SCE, protocol=2)
with open("./CamVid_result/"+"Dirichlet_CamVid_ICCV_Local_36_ACE.txt", "wb") as fp_ACE:
    pickle.dump(res_list_Local_ACE, fp_ACE, protocol=2)

print('ECE Local: ', np.mean(res_list_Local_ECE), np.std(res_list_Local_ECE))
print('MCE Local: ', np.mean(res_list_Local_MCE), np.std(res_list_Local_MCE))
print('SCE Local: ', np.mean(res_list_Local_SCE), np.std(res_list_Local_SCE))
print('ACE Local: ', np.mean(res_list_Local_ACE), np.std(res_list_Local_ACE))


