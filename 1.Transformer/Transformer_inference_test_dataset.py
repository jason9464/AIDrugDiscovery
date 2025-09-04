import numpy as np
from rdkit import Chem
import multiprocessing
import logging
import pandas as pd
import math

import argparse

from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import pyplot

from transformers import BertTokenizer, BertModel
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from pdb import set_trace as bp
from tqdm import tqdm

import torch.utils.data as data

from data_prep import FASTADataset
from model import transformer_regression, transformer_classification


def load_trained_model(model_save_path, gpu, task, model_name='Transformer_IC50'):

    # model
    if task == "classification":
        model = transformer_classification(26, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=2)
    elif task == "regression":
        model = transformer_regression(26, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=1)
    model = model.to(gpu)

    trained_dict = torch.load(model_save_path + model_name, map_location='cpu')

    # version collison handle
    if 'module' in list(trained_dict.keys())[0]:
        tmp_dict = {}
        for key in trained_dict:
            tmp_dict[key[7:]] = trained_dict[key]
        trained_dict = tmp_dict

    model.load_state_dict(trained_dict)

    model.eval()

    return model


def predict(model, test_dataset, gpu, test_data, task): ## 모델 훈련
    predicted_list = []
    predicted_label_list = []
    predicted_probability_list = []
    
    with torch.no_grad():

        for i, data in enumerate(test_dataset):

            X_1, X_2 = data

            X_1 = X_1.view(1, -1)
            X_2 = X_2.view(1, -1)

            mask1 = model.create_padding_mask(X_1, 0)
            mask1 = mask1.to(gpu)

            mask2 = model.create_padding_mask(X_2, 0)
            mask2 = mask2.to(gpu)

            X_1 = X_1.to(gpu)
            X_2 = X_2.to(gpu)

            output = model(X_1, X_2, mask1, mask2)
            
            if task == "classification":
                output = F.softmax(output, dim=1)
                print("[Predicted Probability]", float(torch.max(output, dim=1)[0]))
                predicted_probability_list.append(float(torch.max(output, dim=1)[0]))
                output = torch.max(output, dim=1)[1].item()
                predicted_label_list.append(output)
                print("[Predicted Class]", output)
            elif task == "regression":
                output = pow(10, -output) * pow(10, 6)
                predicted_list.append(output.item())
                print("[Predicted IC50]", output.item())
                print("[True IC50]", test_data.IC50.values[i])
                print("************************************")
    
    if task == "classification":
        pred_dict = {"Antibody": test_data.Antibody, "Antigen": test_data.Antigen, "pred_label": predicted_label_list, "prob": predicted_probability_list}
        pd.DataFrame(pred_dict).to_csv("result_classification.csv")
    elif task == "regression":
        pred_dict = {"Antibody": test_data.Antibody, "Antigen": test_data.Antigen, "pred_IC50": predicted_list, "IC50": test_data.IC50.values}
        pd.DataFrame(pred_dict).to_csv("result_with_log.csv", encoding='utf-8-sig')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="regression", help="[classification, regression]")
    parser.add_argument('--test_path', type=str, default='/home/ngng0274/drug/VirusNet/test_data/')
    parser.add_argument('--model_save_path', type=str,
                        default='/home/ngng0274/drug/VirusNet/trained_model/')

    parser.add_argument('--gpu', type=int, default=-1, help='-1: cpu, else: gpu number')

    opt = parser.parse_args()
    print(opt)

    # Setting
    ## gpu setting
    if opt.gpu == -1:
        gpu = torch.device('cpu')
    else:
        gpu = torch.device('cuda:' + str(opt.gpu))

    MAX_LEN = 60

    ## for dataset, model path
    task, model_save_path, test_path = opt.task, opt.model_save_path, opt.test_path
    test_data = pd.read_excel(test_path + 'Cov-AbDab_data.xlsx') # test file

    # Model Load
    if task == "classification":
        model = load_trained_model(model_save_path, gpu, task, model_name='Transformer_classifier')
        test_dataset = FASTADataset(
            sequences1=test_data.Antibody.to_numpy(),
            sequences2=test_data.Antigen.to_numpy(),
            labels=None,
            auto_padding=MAX_LEN,
            regression=False,
            is_test_data=True
        )
    elif task == "regression":
        model = load_trained_model(model_save_path, gpu, task, model_name='Transformer_regression_with_log')
        test_dataset = FASTADataset(
            sequences1=test_data.Antibody.to_numpy(),
            sequences2=test_data.Antigen.to_numpy(),
            labels=test_data.IC50.values,
            auto_padding=MAX_LEN,
            regression=True,
            is_test_data=True
        )

    # make inference
    ##########################################################################################################################
    
    predict(model, test_dataset, gpu, test_data, task)
