import numpy as np
import pandas as pd
import sys, os
import argparse
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gintransformer import GINTransformer
from utils import *
import torch.utils.data as data

def load_trained_model(model_save_path, gpu, modeling, model_name='model_GINTransformer_BindingDB_30000_0.model'):

    model = modeling().to(gpu)

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


def predicting(model, device, test_dataset):
    predicted_list = []
    DRUG_list = []
    TARGET_list = []
    QED_list = []
    model.eval()

    with torch.no_grad():
        for smiles, target, data, QED in test_dataset:
            mask1 = create_padding_mask(data.target, 0)
            mask1 = mask1.to(device)
            data = data.to(device)
            output = model(data, mask1)
            output = pow(10, -output) * pow(10, 9)
            print(output.item())
            predicted_list.append(output.item())
            DRUG_list.append(smiles)
            TARGET_list.append(target)
            QED_list.append(QED)
            
    print(len(predicted_list))
    pred_dict = {"SMILES_DRUG": DRUG_list, "FASTA_TARGET": TARGET_list, "IC50": predicted_list, "QED": QED_list}
    pd.DataFrame(pred_dict).to_csv("inference/result_generate_9_26_30000_full_log6.csv")
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--model", type=int, default=4, help="[GINConvNet, GATNet, GAT_GCN, GCNNet, GINTransformer]")
    parser.add_argument('--model_save_path', type=str,
                        default='trained_model/')

    parser.add_argument('--test_dataset', type=str,
                        default='generated_molecules_9_26.csv', help='Test Dataset')
    parser.add_argument('--test_dataset_save_path', type=str,
                        default='test_data/')
    
    parser.add_argument('--gpu', type=int, default=-1, help='-1: cpu, else: gpu number')

    opt = parser.parse_args()
    print(opt)
    
    # Setting
    ## gpu setting
    if opt.gpu == -1:
        gpu = torch.device('cpu')
    else:
        gpu = torch.device('cuda:' + str(opt.gpu))
    
    ## for dataset, model path
    model_idx, model_save_path = opt.model, opt.model_save_path
    test_dataset, test_dataset_save_path = opt.test_dataset, opt.test_dataset_save_path
    
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GINTransformer][model_idx]
    model_st = modeling.__name__
    model = load_trained_model(model_save_path, gpu, modeling, model_name='model_GINTransformer_BindingDB0.0001_700_30000_log6.model')

    test_df = pd.read_csv(test_dataset_save_path + test_dataset)

    TEST_BATCH_SIZE = 32
    
    # test_df = test_df[test_df['SMILES_DRUG'] != '[Cl-].[K+]']
    test_df = test_df[~test_df['SMILES_DRUG'].str.contains('^\w+$')]
    test_df = test_df[~test_df['SMILES_DRUG'].str.contains('^\[\w+[+-]*\d*\]$')]
    test_df = test_df[~test_df['SMILES_DRUG'].str.contains('^\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]$')]
    test_df = test_df[~test_df['SMILES_DRUG'].str.contains('^\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]$')]
    test_df = test_df[~test_df['SMILES_DRUG'].str.contains('^\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]$')]
    test_df = test_df[~test_df['SMILES_DRUG'].str.contains('^\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]\.\[\w+[+-]*\d*\]$')]
    
    test_dataset = TestDataset(test_df)
    
    predicting(model, gpu, test_dataset)
