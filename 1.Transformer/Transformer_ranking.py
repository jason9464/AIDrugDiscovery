import numpy as np
import matplotlib.pyplot  as plt

from rdkit import Chem
import multiprocessing
import logging
import pandas as pd
import math
import time
import datetime

import argparse
import os
from sklearn.metrics import mean_squared_error
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
from model import transformer_rank_added
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def train(args, gpu, criterion1, criterion2, dataset, model_save_path, MAX_LEN, is_save=False): ## 모델 훈련
    num_epochs, lr, reg = args.num_epochs, args.lr, args.reg
    
    save_dir = "result_rank/{}_lr{}_epoch{}_reg{}".format("rank", lr, num_epochs, reg)
    os.makedirs(save_dir, exist_ok=True)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)
    
    fold_train_loss, fold_test_loss = [], []
    
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset, dataset.labels)):
        train_dataset = torch.utils.data.dataset.Subset(dataset, train_index)
        test_dataset = torch.utils.data.dataset.Subset(dataset, test_index)

        train_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        model = transformer_rank_added(21, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=3, MAX_LEN=MAX_LEN)
        model.to(gpu)
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=lr, weight_decay=reg)
        # optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=reg)
        for epoch in range(num_epochs):
            train_loss_list = []
            test_loss_list = []
            output_list, y_list = [], []
            antibody_list, antigen_list = [], []
            
            with tqdm(total=len(train_loader)) as pbar:

                for batch_idx, batch in enumerate(train_loader):

                    batch_X_1, batch_X_2, batch_feats1, batch_feats2, batch_y, batch_antibody, batch_antigen = batch

                    # Convert numpy arrays to torch tensors

                    batch_y = batch_y.to(gpu)
                    mask1 = model.create_padding_mask(batch_X_1, 0) ## 항체 길이에 맞는 padding mask 생성
                    mask1 = mask1.to(gpu)

                    batch_y = batch_y.to(gpu)
                    mask2 = model.create_padding_mask(batch_X_2, 0) ## 표적 단백질 길이에 맞는 padding mask 생성
                    mask2 = mask2.to(gpu)

                    batch_X_1 = batch_X_1.to(gpu)
                    batch_X_2 = batch_X_2.to(gpu)
                    batch_feats1 = batch_feats1.to(gpu)
                    batch_feats2 = batch_feats2.to(gpu)

                    a, b, c = torch.FloatTensor([1] * batch_X_1.size(0)), torch.FloatTensor([2] * batch_X_1.size(0)), torch.FloatTensor([3] * batch_X_1.size(0))
                    value = torch.stack([a, b, c]).T
                    value = value.to(gpu)
                    
                    # Forward Pass
                    model.train()

                    output = model(batch_X_1, batch_X_2, batch_feats1, batch_feats2, mask1, mask2)
                    
                    A = F.softmax(output, dim=1)
                    A = (A * value).sum(1).squeeze()

                    train_loss = torch.sum(criterion1(A, batch_y + 1.) + criterion2(output, batch_y))

                    train_loss_list.append(train_loss.item())

                    # Backward and optimize
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    
                    pbar.update(1)

                # 기존
                with torch.no_grad():
                    model.eval()
                    
                    for X_1, X_2, feats1, feats2, y, antibody, antigen in tqdm(test_loader):

                        mask1 = model.create_padding_mask(X_1, 0)
                        mask1 = mask1.to(gpu)

                        mask2 = model.create_padding_mask(X_2, 0)
                        mask2 = mask2.to(gpu)

                        X_1 = X_1.to(gpu)
                        X_2 = X_2.to(gpu)
                        feats1 = feats1.to(gpu)
                        feats2 = feats2.to(gpu)

                        y = y.to(gpu)

                        a, b, c = torch.FloatTensor([1] * X_1.size(0)), torch.FloatTensor([2] * X_1.size(0)), torch.FloatTensor([3] * X_1.size(0))
                        value = torch.stack([a, b, c]).T
                        value = value.to(gpu)

                        output = model(X_1, X_2, feats1, feats2, mask1, mask2)
                        
                        A = F.softmax(output, dim=1)
                        A = (A * value).sum(1).squeeze()
                        
                        test_loss = torch.sum(criterion1(A, y + 1.))

                        output_list.extend(A-1.)
                        y_list.extend(y)

                        antibody_list.extend(antibody)
                        antigen_list.extend(antigen)
                    
                    output_list = torch.stack(output_list)
                    output_list = torch.squeeze(output_list)
                    y_list = torch.stack(y_list)
                    y_list = torch.squeeze(y_list)
                  
                    # output_list = F.softmax(output_list, dim=1)
                    # prob_list = torch.max(output_list, dim=1)[0]
                    # output_list = torch.max(output_list, dim=1)[1]

                output_list = torch.clamp(output_list, 0, 2).round()

                pred_dict = {"CDRH3": antibody_list, "CDRL3": antigen_list, "pred_label": [int(i) for i in output_list], "Class": [int(i) for i in y_list]}
                pd.DataFrame(pred_dict).to_csv("classification_testdataset_result.csv")
                train_loss = sum(train_loss_list)/len(train_loss_list)
                acc = float((y_list == output_list).sum()) / y_list.size(0)
                print("Fold: {}, Epoch: {}, train_loss: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}".format(fold, epoch, train_loss, test_loss, acc))

        final_train_loss = train_loss
        fold_train_loss.append(final_train_loss)
        
        final_test_acc = acc
        fold_test_loss.append(final_test_acc)
        
        precision = precision_score(y_list.data.cpu().numpy(), output_list.data.cpu().numpy(), average='macro')
        recall = recall_score(y_list.data.cpu().numpy(), output_list.data.cpu().numpy(), average='macro')
        f1 = f1_score(y_list.data.cpu().numpy(), output_list.data.cpu().numpy(), average='macro')
        print("Fold: {}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(fold, precision, recall, f1))
        
        torch.save(model.state_dict(), '{}/model_{}.pt'.format(save_dir, fold))
        
    mean_acc = sum(fold_test_loss)/len(fold_test_loss)
    std = np.std(np.array(fold_test_loss))
    print("Mean acc: {:.4f}".format(mean_acc))
    print("std: {:.4f}".format(std))
    f = open("{}/result.txt".format(save_dir), 'w')
    f.write("Fold num: {}, epochs: {}\n".format(5, num_epochs))
    f.write("Mean acc: {:.4f}\n".format(mean_acc))
    f.write("Std: {:.4f}\n".format(std.item()))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0001) ## learning rate
    parser.add_argument('--reg', type=float, default=0) ## weight decay
    parser.add_argument('--batch_size', type=int, default=512) ## batch_size
    parser.add_argument('--num_epochs', type=int, default=200) ## num_epochs
    parser.add_argument('--early_stop', type=int, default=50) ## early_stop 

    parser.add_argument('--model_save', type=int,
                        default=0, help='0: false, 1:true')
    parser.add_argument('--model_save_path', type=str,
                        default='/home/ngng0274/drug/VirusNet/trained_model/')

    parser.add_argument('--gpu', type=int, default=2, help='0,1,2,3') ## 사용할 gpu

    opt = parser.parse_args()
    print(opt)

    gpu = torch.device('cuda:' + str(opt.gpu))


    # for model model_save_path
    model_save_path, model_save = opt.model_save_path, opt.model_save
    model_save_path += 'Transformer_cls_Cov_1214'

    if model_save == 0:
        model_save = False
    elif model_save == 1:
        model_save = True
    else:
        print("check model save argument")
        exit()

    # for train
    MAX_LEN = 30
    lr, batch_size, reg = opt.lr, opt.batch_size, opt.reg

    ## dataset
    df = pd.read_csv('SARS-CoV2_Omicron-BA1_IC50_20230110.csv', encoding='cp949') # dataset file here

    dataset = FASTADataset(
        sequences1=df.CDRH3,
        sequences2=df.CDRL3,
        labels=df.label.to_numpy(),
        auto_padding=MAX_LEN,
        regression=False
    )
    criterion1 = nn.MSELoss(reduce=False)
    criterion2 = torch.nn.CrossEntropyLoss()
    
    start = time.time()
    
    train(opt, gpu, criterion1, criterion2, dataset, model_save_path, MAX_LEN, is_save=model_save)
    
    end = time.time()

    result_list = str(datetime.timedelta(seconds=(end - start))).split(".")
    print(result_list[0])
