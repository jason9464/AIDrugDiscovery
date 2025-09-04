import numpy as np
from rdkit import Chem
import multiprocessing
import logging
import pandas as pd
import math

import argparse

from sklearn.model_selection import train_test_split

import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from pdb import set_trace as bp
from tqdm import tqdm

import torch.utils.data as data

from data_prep_added import FASTADataset
from model import transformer_classification_added


def load_trained_model(model_save_path, gpu):

    # model
    model = transformer_classification_added(21, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=2)

    model = model.to(gpu)

    trained_dict = torch.load(model_save_path, map_location='cpu')

    # version collison handle
    if 'module' in list(trained_dict.keys())[0]:
        tmp_dict = {}
        for key in trained_dict:
            tmp_dict[key[7:]] = trained_dict[key]
        trained_dict = tmp_dict

    model.load_state_dict(trained_dict)

    model.eval()

    return model

def predict(model, test_loader, gpu): ## 모델 훈련
    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            X_1, X_2, feats1, feats2 = batch

            mask1 = model.create_padding_mask(X_1, 0)
            mask1 = mask1.to(gpu)

            mask2 = model.create_padding_mask(X_2, 0)
            mask2 = mask2.to(gpu)

            X_1 = X_1.to(gpu)
            X_2 = X_2.to(gpu)
            feats1 = feats1.to(gpu)
            feats2 = feats2.to(gpu)

            output = model(X_1, X_2, feats1, feats2, mask1, mask2)

            output = F.softmax(output, dim=1)
            print("[Predicted Probability]", float(torch.max(output, dim=1)[0]))
            output = int(torch.max(output, dim=1)[1])
            print("[Predicted Class]", output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_save_path', type=str,
                        default='trained_model/')

    parser.add_argument('--CDRH3', type=str,
                        default='AAPHCNSTSCYDAFDI', help='CDRH3 Sequence')
    parser.add_argument('--CDRL3', type=str,
                        default='QQYGSSPWT', help='CDRL3 Sequence')
    
    parser.add_argument('--gpu', type=int, default=-1, help='-1: cpu, else: gpu number')

    opt = parser.parse_args()
    print(opt)

    # Setting
    ## gpu setting
    if opt.gpu == -1:
        gpu = torch.device('cpu')
    else:
        gpu = torch.device('cuda:' + str(opt.gpu))

    MAX_LEN = 30

    ## for dataset, model path
    model_save_path = opt.model_save_path
    model_save_path += 'Transformer_classifier_full_dataset'
    
    CDRH3, CDRL3 = opt.CDRH3, opt.CDRL3

    test_data = {'CDRH3': [CDRH3], 'CDRL3': [CDRL3]}
    test_data = pd.DataFrame(test_data)

    # Model Load
    model = load_trained_model(model_save_path, gpu)
    test_dataset = FASTADataset(
        sequences1=test_data.CDRH3.to_numpy(),
        sequences2=test_data.CDRL3.to_numpy(),
        labels=None,
        auto_padding=MAX_LEN,
        regression=False,
        is_test_data=True
    )

    test_loader = data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # make inference
    ##########################################################################################################################
    
    predict(model, test_loader, gpu)
