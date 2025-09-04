import numpy as np
from rdkit import Chem
import multiprocessing
import logging
import pandas as pd

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

def evaluate_category(model, test_loader, gpu, criterion, test=False): ## dataset에 대한 evaluation

    output_list, y_list, prob_list = [], [], []

    antibody_list, antigen_list = [], []
    
    total_loss = 0.

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

            output = model(X_1, X_2, feats1, feats2, mask1, mask2)
            output_list.extend(output)
            y_list.extend(y)

            antibody_list.extend(antibody)
            antigen_list.extend(antigen)
            
            total_loss += criterion(output, y).data.cpu().numpy()

        output_list = torch.stack(output_list)
        y_list = torch.stack(y_list)
        y_list = torch.squeeze(y_list)

        output_list = F.softmax(output_list, dim=1)
        prob_list = torch.max(output_list.data.cpu(), dim=1)[0]
        output_list = torch.max(output_list, dim=1)[1]

        ACC = float((y_list == output_list).sum()) / y_list.size(0)

    return total_loss / len(test_loader), ACC

def print_result_category(epoch, num_epochs, train_loss, ACC_train): ## 결과 출력 함수
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train ACC: {:.4f}'.format(
            epoch, num_epochs, train_loss, ACC_train))

def train(args, gpu, criterion, optimizer, model, train_loader, is_save=False): ## 모델 훈련
    train_loss_arr = []

    early_stop, early_stop_max, num_epochs = 0., args.early_stop, args.num_epochs

    for epoch in range(num_epochs):

        epoch_loss = 0.

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
                
                # Forward Pass
                model.train()
                
                output = model(batch_X_1, batch_X_2, batch_feats1, batch_feats2, mask1, mask2)
                train_loss = criterion(output, batch_y)

                with torch.no_grad():
                    epoch_loss += train_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                pbar.update(1)
            train_loss_arr.append(epoch_loss / len(train_loader))

            if is_save:
                torch.save(model.state_dict(), model_save_path)
            
            # 기존
            if epoch % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    ## 각 dataset에 대한 loss 계산
                    train_loss, ACC_train = evaluate_category(
                        model, train_loader, gpu, criterion)

                    print("=" * 100)
                    print_result_category(epoch, num_epochs, train_loss, ACC_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0001) ## learning rate
    parser.add_argument('--reg', type=float, default=0) ## weight decay
    parser.add_argument('--batch_size', type=int, default=512) ## batch_size
    parser.add_argument('--num_epochs', type=int, default=170) ## num_epochs
    parser.add_argument('--early_stop', type=int, default=50) ## --early_stop

    parser.add_argument('--model_save', type=int,
                        default=1, help='0: false, 1:true')
    parser.add_argument('--model_save_path', type=str,
                        default='/home/ngng0274/drug/VirusNet/trained_model/')

    parser.add_argument('--gpu', type=int, default=2, help='0,1,2,3') ## 사용할 gpu

    opt = parser.parse_args()
    print(opt)

    gpu = torch.device('cuda:' + str(opt.gpu))

    # for model model_save_path
    model_save_path, model_save = opt.model_save_path, opt.model_save
    model_save_path += 'Transformer_classifier_full_170'

    if model_save == 0:
        model_save = False
    elif model_save == 1:
        model_save = True
    else:
        print("check model save argument")
        exit()

    MAX_LEN = 30
    lr, batch_size, reg = opt.lr, opt.batch_size, opt.reg

    ## dataset
    df = pd.read_csv('SARS-CoV2_Omicron-BA1_IC50_20230118.csv') # dataset file here

    ## 각 dataset에 대한 전처리
    train_dataset = FASTADataset(
        sequences1=df.CDRH3.to_numpy(),
        sequences2=df.CDRL3.to_numpy(),
        labels=df.label.to_numpy(),
        auto_padding=MAX_LEN,
        regression=False
    )
    ## 각 dataset에 대한 dataloader 생성
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    
    model = transformer_classification_added(21, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=2)
    model.to(gpu)

    # nSamples = [1041, 1490]
    # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    # normedWeights = torch.FloatTensor(normedWeights).to(gpu)
    # criterion = torch.nn.CrossEntropyLoss(normedWeights)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=lr, weight_decay=reg)
    
    train(opt, gpu, criterion, optimizer, model, train_loader, is_save=model_save)
