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

def load_trained_model(model_save_path, gpu, modeling, model_name='model_GINTransformer_davis.model'):

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


def predicting(model, device, test_data):
    model.eval()
    total_preds = torch.Tensor()

    with torch.no_grad():
        for data in test_data:
            mask1 = create_padding_mask(data.target, 0)
            mask1 = mask1.to(device)
            data = data.to(device)
            output = model(data, mask1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)

    return total_preds.numpy().flatten()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    
    parser.add_argument("--model", type=int, default=4, help="[GINConvNet, GATNet, GAT_GCN, GCNNet, GINTransformer]")
    parser.add_argument('--model_save_path', type=str,
                        default='trained_model/')

    parser.add_argument('--drug', type=str,
                        default='CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N', help='Drug Sequence')
    parser.add_argument('--target', type=str,
                        default='MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL', help='Target Protein Sequence')
    
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
    drug, target = opt.drug, opt.target
    
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GINTransformer][model_idx]
    model_st = modeling.__name__
    model = load_trained_model(model_save_path, gpu, modeling, model_name='model_GINTransformer_davis.model')

    test_data = {'SMILES_DRUG': [drug], 'FASTA_TARGET': [target]}
    test_data = pd.DataFrame(test_data)

    TEST_BATCH_SIZE = 32
    LOG_INTERVAL = 20
    
    test_data = TestDataset(test_data)
    
    P = predicting(model, gpu, test_data)
    
    print("[Predicted pKd]", P)
    print("[Predicted Kd]", pow(10, -P) * pow(10, 9))