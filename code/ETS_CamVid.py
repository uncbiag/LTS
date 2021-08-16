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
from scipy import optimize
from sklearn.isotonic import IsotonicRegression
from probability_measure_CamVid import Calculate_ECE, Calculate_MCE, Calculate_SCE, Calculate_ACE
import pickle
from scipy import optimize


def mse_t(t, *args):
    ## find optimal temperature with MSE loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.exp(logit) / n[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_t(t, *args):
    ## find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.clip(np.exp(logit) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def mse_w(w, *args):
    ## find optimal weight coefficients with MSE loss function

    p0, p1, p2, label = args
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    p = p / np.sum(p, 1)[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, label = args
    p = (w[0] * p0 + w[1] * p1 + w[2] * p2)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


##### Ftting Temperature Scaling
def temperature_scaling(logit, label, loss):
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
        t = optimize.minimize(ll_t, 1.0, args=(logit, label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    if loss == 'mse':
        t = optimize.minimize(mse_t, 1.0, args=(logit, label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    t = t.x
    return t


##### Ftting Enseble Temperature Scaling
def ensemble_scaling(logit, label, loss, t, n_class):
    p1 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    logit = logit / t
    p0 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p2 = np.ones_like(p0) / n_class

    bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)

    def my_constraint_fun(x):
        return np.sum(x) - 1

    constraints = {"type": "eq", "fun": my_constraint_fun, }
    if loss == 'ce':
        w = optimize.minimize(ll_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP', constraints=constraints,
                              bounds=bnds_w, tol=1e-12, options={'disp': True})
    if loss == 'mse':
        w = optimize.minimize(mse_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP', constraints=constraints,
                              bounds=bnds_w, tol=1e-12, options={'disp': True})
    w = w.x
    return w


def ets_calibrate(logit,label,logit_eval,n_class,loss):
    t = temperature_scaling(logit,label,loss='ce') # loss can change to 'ce'
    print("temperature = " +str(t))
    w = ensemble_scaling(logit,label,'ce',t,n_class)
    print("weight = " +str(w))


    p1 = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]
    logit_eval = logit_eval/t
    p0 = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]
    p2 = np.ones_like(p0)/n_class
    p = w[0]*p0 + w[1]*p1 +w[2]*p2
    return p


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

print("merge individual cases")
all_probs = None
all_logits = None
all_labels = None
for i, (val_image, val_logits, val_labels, val_preds, val_boundary) in enumerate(TIRAMISU_val_dataloader):
    test_probs = torch.softmax(val_logits, dim=1).detach().squeeze().cpu().numpy()
    val_label_array = val_labels.detach().squeeze().cpu().numpy()
    prob_img_array_select = np.transpose(test_probs.reshape((12, -1)))
    logits_img_array_select = np.transpose(val_logits.detach().squeeze().cpu().numpy().reshape((12, -1)))
    val_label_array_select = val_label_array
    val_label_array_select_onehot = np.eye(12)[val_label_array_select].reshape((-1, 12))
    if all_probs is None:
        all_probs = prob_img_array_select
        all_logits = logits_img_array_select
        all_labels = val_label_array_select_onehot
    else:
        all_probs = np.concatenate((all_probs, prob_img_array_select), axis=0)
        all_logits = np.concatenate((all_logits, logits_img_array_select), axis=0)
        all_labels = np.concatenate((all_labels, val_label_array_select_onehot), axis=0)


t = temperature_scaling(all_logits, all_labels, loss='ce')
print("temperature = " +str(t))
w = ensemble_scaling(all_logits, all_labels, 'ce', t, 12)
print("weight = " +str(w))

res_list_All_ECE = []
res_list_All_MCE = []
res_list_All_SCE = []
res_list_All_ACE = []
res_list_Boundary_ECE = []
res_list_Boundary_MCE = []
res_list_Boundary_SCE = []
res_list_Boundary_ACE = []


for ind, (test_image, test_logits, test_labels, test_preds, test_boundary) in enumerate(TIRAMISU_test_dataloader):
    print(ind)

    image_shape = test_logits.squeeze().shape

    test_t_probs = torch.softmax(test_logits/torch.from_numpy(t).cuda(), dim=1).detach().squeeze().cpu().numpy()
    test_ut_probs = torch.softmax(test_logits, dim=1).detach().squeeze().cpu().numpy()
    new_probs = test_t_probs*w[0] + test_ut_probs*w[1] + (1.0/12.0) * w[2]
    prob_img_array = np.max(new_probs, axis=0)/np.sum(new_probs, axis=0)


    pred_img_array = test_preds.squeeze().cpu().numpy()
    gt_img_array   = test_labels.squeeze().cpu().numpy()
    boundary_img_array = test_boundary.squeeze().cpu().numpy()


    res_list_All_ECE.append(Calculate_ECE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
    res_list_All_MCE.append(Calculate_MCE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
    res_list_All_SCE.append(Calculate_SCE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
    res_list_All_ACE.append(Calculate_ACE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
    res_list_Boundary_ECE.append(Calculate_ECE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=True, n_bins=10))
    res_list_Boundary_MCE.append(Calculate_MCE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=True, n_bins=10))
    res_list_Boundary_SCE.append(Calculate_SCE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=True, n_bins=10))
    res_list_Boundary_ACE.append(Calculate_ACE(confidence=prob_img_array, prediction=pred_img_array, gt=gt_img_array, boundary=boundary_img_array, boundary_on=True, n_bins=10))


with open("./CamVid_result/"+"ETS_CamVid_ICCV_All_ECE.txt", "wb") as fp_ECE:
    pickle.dump(res_list_All_ECE, fp_ECE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_All_MCE.txt", "wb") as fp_MCE:
    pickle.dump(res_list_All_MCE, fp_MCE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_All_SCE.txt", "wb") as fp_SCE:
    pickle.dump(res_list_All_SCE, fp_SCE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_All_ACE.txt", "wb") as fp_ACE:
    pickle.dump(res_list_All_ACE, fp_ACE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_Boundary_ECE.txt", "wb") as fp_B_ECE:
    pickle.dump(res_list_Boundary_ECE, fp_B_ECE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_Boundary_MCE.txt", "wb") as fp_B_MCE:
    pickle.dump(res_list_Boundary_MCE, fp_B_MCE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_Boundary_SCE.txt", "wb") as fp_B_SCE:
    pickle.dump(res_list_Boundary_SCE, fp_B_SCE, protocol=2)
with open("./CamVid_result/"+"ETS_CamVid_ICCV_Boundary_ACE.txt", "wb") as fp_B_ACE:
    pickle.dump(res_list_Boundary_ACE, fp_B_ACE, protocol=2)


print('ECE All: ', np.mean(res_list_All_ECE), np.std(res_list_All_ECE))
print('MCE All: ', np.mean(res_list_All_MCE), np.std(res_list_All_MCE))
print('SCE All: ', np.mean(res_list_All_SCE), np.std(res_list_All_SCE))
print('ACE All: ', np.mean(res_list_All_ACE), np.std(res_list_All_ACE))
print('ECE Boundary: ', np.mean(res_list_Boundary_ECE), np.std(res_list_Boundary_ECE))
print('MCE Boundary: ', np.mean(res_list_Boundary_MCE), np.std(res_list_Boundary_MCE))
print('SCE Boundary: ', np.mean(res_list_Boundary_SCE), np.std(res_list_Boundary_SCE))
print('ACE Boundary: ', np.mean(res_list_Boundary_ACE), np.std(res_list_Boundary_ACE))


