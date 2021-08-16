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
import sys

def Calculate_ECE(confidence, prediction, gt, boundary, boundary_on = False, n_bins=10):
    '''
    Expected Calibration Error
    :param confidence: probability map of segmentation
    :param prediction: prediction map of segmentation
    :param gt: groud-truth segmentation map
    :param boundary: boundary areas of segmentation map
    :param boundary_on: whether to evaluate on boundary or all region (gt != 100).
                        All region are defined by the regions have manual segmentations and the boundary areas.
    :param n_bins: how many bins to evaluate
    :return: ece value
    '''

    bin_boundaries = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0

    accuracy = (prediction == gt)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if boundary_on:
            in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (boundary != 0)
            prop_in_bin = in_bin[boundary != 0].astype(float).mean()
        else:
            in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (gt != 100)
            prop_in_bin = in_bin[gt != 100].astype(float).mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def Calculate_MCE(confidence, prediction, gt, boundary, boundary_on = False, n_bins=10):
    '''
    Maximum Calibration Error
    :param confidence: probability map of segmentation
    :param prediction: prediction map of segmentation
    :param gt: groud-truth segmentation map
    :param boundary: boundary areas of segmentation map
    :param boundary_on: whether to evaluate on boundary or all region (gt != 100).
                        All region are defined by the regions have manual segmentations and the boundary areas.
    :param n_bins: how many bins to evaluate
    :return: mce value
    '''
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0
    accuracy = (prediction == gt)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if boundary_on:
            in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (boundary != 0)
            prop_in_bin = in_bin[boundary != 0].astype(float).mean()
        else:
            in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (gt != 100)
            prop_in_bin = in_bin[gt != 100].astype(float).mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            mce = max(abs(avg_confidence_in_bin - accuracy_in_bin), mce)

    return mce

def Calculate_SCE(confidence, prediction, gt, boundary, boundary_on = False, n_bins=10):
    '''
    Static Calibration Error
    :param confidence: probability map of segmentation
    :param prediction: prediction map of segmentation
    :param gt: groud-truth segmentation map
    :param boundary: boundary areas of segmentation map
    :param boundary_on: whether to evaluate on boundary or all region (gt != 100).
                        All region are defined by the regions have manual segmentations and the boundary areas.
    :param n_bins: how many bins to evaluate
    :return: sce value
    '''
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    sce = 0
    accuracy = (prediction == gt)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        for label_ind in range(12):
            if boundary_on:
                in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (boundary != 0) * (prediction == label_ind)
                prop_in_bin = in_bin[boundary != 0].astype(float).mean()
            else:
                in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (gt != 100) * (prediction == label_ind)
                prop_in_bin = in_bin[gt != 100].astype(float).mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidence[in_bin].mean()

                sce += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return sce

def Calculate_ACE(confidence, prediction, gt, boundary, boundary_on = False, n_bins=10):
    '''
    Adaptive Calibration Error
    :param confidence: probability map of segmentation
    :param prediction: prediction map of segmentation
    :param gt: groud-truth segmentation map
    :param boundary: boundary areas of segmentation map
    :param boundary_on: whether to evaluate on boundary or all region (gt != 100).
                        All region are defined by the regions have manual segmentations and the boundary areas.
    :param n_bins: how many bins to evaluate
    :return: ace value
    '''
    bin_boundaries = [np.percentile(confidence, (float(i)*100.0/float(n_bins))) for i in range(1, n_bins)]
    bin_boundaries = [0.0] + bin_boundaries + [1.0]
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ace = 0
    accuracy = (prediction == gt)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        for label_ind in range(12):
            if boundary_on:
                in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (boundary != 0) * (prediction == label_ind)
                prop_in_bin = in_bin[boundary != 0].astype(float).mean()
            else:
                in_bin = (confidence > bin_lower) * (confidence <= bin_upper) * (gt != 100) * (prediction == label_ind)
                prop_in_bin = in_bin[gt != 100].astype(float).mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidence[in_bin].mean()

                ace += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ace



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int, help='index of used GPU')
    parser.add_argument('--model_name', default='LTS', type=str, help='model name: IBTS, LTS, TS, UN')
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

        test_image, test_logits = test_image.to('cuda'), test_logits.to('cuda')
        logits_cali = calibration_model(test_logits, test_image, args)
        test_probs = torch.softmax(logits_cali, dim=1)
        prob_img_array = torch.max(test_probs, dim=1)[0].detach().squeeze().cpu().numpy()

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

    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_All_ECE.txt", "wb") as fp_ECE:
        pickle.dump(res_list_All_ECE, fp_ECE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_All_MCE.txt", "wb") as fp_MCE:
        pickle.dump(res_list_All_MCE, fp_MCE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_All_SCE.txt", "wb") as fp_SCE:
        pickle.dump(res_list_All_SCE, fp_SCE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_All_ACE.txt", "wb") as fp_ACE:
        pickle.dump(res_list_All_ACE, fp_ACE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Boundary_ECE.txt", "wb") as fp_B_ECE:
        pickle.dump(res_list_Boundary_ECE, fp_B_ECE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Boundary_MCE.txt", "wb") as fp_B_MCE:
        pickle.dump(res_list_Boundary_MCE, fp_B_MCE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Boundary_SCE.txt", "wb") as fp_B_SCE:
        pickle.dump(res_list_Boundary_SCE, fp_B_SCE, protocol=2)
    with open("./CamVid_result/"+model_name+"_CamVid_ICCV_Boundary_ACE.txt", "wb") as fp_B_ACE:
        pickle.dump(res_list_Boundary_ACE, fp_B_ACE, protocol=2)

    print('ECE All: ', np.mean(res_list_All_ECE), np.std(res_list_All_ECE))
    print('MCE All: ', np.mean(res_list_All_MCE), np.std(res_list_All_MCE))
    print('SCE All: ', np.mean(res_list_All_SCE), np.std(res_list_All_SCE))
    print('ACE All: ', np.mean(res_list_All_ACE), np.std(res_list_All_ACE))
    print('ECE Boundary: ', np.mean(res_list_Boundary_ECE), np.std(res_list_Boundary_ECE))
    print('MCE Boundary: ', np.mean(res_list_Boundary_MCE), np.std(res_list_Boundary_MCE))
    print('SCE Boundary: ', np.mean(res_list_Boundary_SCE), np.std(res_list_Boundary_SCE))
    print('ACE Boundary: ', np.mean(res_list_Boundary_ACE), np.std(res_list_Boundary_ACE))


