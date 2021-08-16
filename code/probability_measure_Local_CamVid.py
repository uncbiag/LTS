import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import glob
from PIL import Image
from calibration_models import *
from Tiramisu_calibration_Dataset import *
from torch.utils.data import DataLoader
import pickle
import argparse
import os
import random
from probability_measure_CamVid import Calculate_ECE, Calculate_MCE, Calculate_SCE, Calculate_ACE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int, help='index of used GPU')
    parser.add_argument('--model_name', default='New', type=str, help='model name: IBTS, LTS, TS, UN')
    parser.add_argument('--patch_len', default=36, type=int, help='local patch length')
    args = parser.parse_args()
    ## CamVid experiment

    model_name = str(args.model_name)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if model_name == 'IBTS':
        model_state_dict = torch.load('YOUR_PATH_TO_SAVED_PARAMETERS')
        calibration_model = IBTS_CamVid_With_Image()
    elif model_name == 'LTS':
        model_state_dict = torch.load('YOUR_PATH_TO_SAVED_PARAMETERS')
        calibration_model = LTS_CamVid_With_Image()
    elif model_name == 'TS':
        model_state_dict = torch.load('YOUR_PATH_TO_SAVED_PARAMETERS')
        calibration_model = Temperature_Scaling()
    else:
        raise ValueError('Wrong model name!')


    TIRAMISU_test = TIRAMISU_CALIBRATION(total_logits_list, 'test')
    TIRAMISU_test_dataloader = DataLoader(TIRAMISU_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    calibration_model.cuda()
    calibration_model.load_state_dict(model_state_dict, strict=False)
    calibration_model.eval()

    ## define seed and local patch size (patch_len*2)
    random.seed(10)
    patch_len = int(args.patch_len)

    res_list_Local_ECE = []
    res_list_Local_MCE = []
    res_list_Local_SCE = []
    res_list_Local_ACE = []

    for ind, (test_image, test_logits, test_labels, test_preds, test_boundary) in enumerate(TIRAMISU_test_dataloader):
        print(ind)

        test_image, test_logits = test_image.to('cuda'), test_logits.to('cuda')
        logits_cali = calibration_model(test_logits, test_image)
        test_probs = torch.softmax(logits_cali, dim=1)
        prob_img_array = torch.max(test_probs, dim=1)[0].detach().squeeze().cpu().numpy()

        pred_img_array = test_preds.squeeze().cpu().numpy()
        gt_img_array   = test_labels.squeeze().cpu().numpy()
        boundary_img_array = test_boundary.squeeze().cpu().numpy()

        for lo_ind in range(10):
            random_center = (random.randint(84, pred_img_array.shape[0]-84), random.randint(84, pred_img_array.shape[1]-84))

            patch_prob_img_array = prob_img_array[random_center[0]-patch_len:random_center[0]+patch_len, random_center[1]-patch_len:random_center[1]+patch_len]
            patch_pred_img_array = pred_img_array[random_center[0]-patch_len:random_center[0]+patch_len, random_center[1]-patch_len:random_center[1]+patch_len]
            patch_gt_img_array = gt_img_array[random_center[0]-patch_len:random_center[0]+patch_len, random_center[1]-patch_len:random_center[1]+patch_len]

            res_list_Local_ECE.append(Calculate_ECE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
            res_list_Local_MCE.append(Calculate_MCE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
            res_list_Local_SCE.append(Calculate_SCE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))
            res_list_Local_ACE.append(Calculate_ACE(confidence=patch_prob_img_array, prediction=patch_pred_img_array, gt=patch_gt_img_array, boundary=boundary_img_array, boundary_on=False, n_bins=10))

    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Local_"+str(patch_len)+"_ECE.txt", "wb") as fp_ECE:
        pickle.dump(res_list_Local_ECE, fp_ECE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Local_"+str(patch_len)+"_MCE.txt", "wb") as fp_MCE:
        pickle.dump(res_list_Local_MCE, fp_MCE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Local_"+str(patch_len)+"_SCE.txt", "wb") as fp_SCE:
        pickle.dump(res_list_Local_SCE, fp_SCE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Local_"+str(patch_len)+"_ACE.txt", "wb") as fp_ACE:
        pickle.dump(res_list_Local_ACE, fp_ACE, protocol=2)

    print('ECE Local: ', np.mean(res_list_Local_ECE), np.std(res_list_Local_ECE))
    print('MCE Local: ', np.mean(res_list_Local_MCE), np.std(res_list_Local_MCE))
    print('SCE Local: ', np.mean(res_list_Local_SCE), np.std(res_list_Local_SCE))
    print('ACE Local: ', np.mean(res_list_Local_ACE), np.std(res_list_Local_ACE))


