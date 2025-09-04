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

from data_prep import FASTADataset
from model import transformer_regression


def evaluate_category(model, test_loader, gpu, criterion): ## dataset에 대한 evaluation

    output_list, y_list = [], []

    total_loss = 0.

    with torch.no_grad():

        for X_1, X_2, y in tqdm(test_loader):

            mask1 = model.create_padding_mask(X_1, 0)
            mask1 = mask1.to(gpu)

            mask2 = model.create_padding_mask(X_2, 0)
            mask2 = mask2.to(gpu)

            X_1 = X_1.to(gpu)
            X_2 = X_2.to(gpu)

            y = y.to(gpu)

            output = model(X_1, X_2, mask1, mask2)

            output_list.extend(output.data.cpu())
            y_list.extend(y.data.cpu())
        
        output_list = torch.stack(output_list)
        y_list = torch.stack(y_list)
        y_list = torch.squeeze(y_list)

        rmse = math.sqrt(mean_squared_error(output_list, y_list))

    return rmse

def print_result_category(epoch, num_epochs, train_loss, val_loss, is_improved=False): ## 결과 출력 함수
    if is_improved:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f} *'.format(
            epoch, num_epochs, train_loss, val_loss))
    else:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(
            epoch, num_epochs, train_loss, val_loss))

def train(args, gpu, criterion, optimizer, model, train_loader, valid_loader, test_loader, model_save_path, is_save=False): ## 모델 훈련
    train_loss_arr = []
    val_loss_arr = []
    test_loss_arr = []

    best_loss, final_loss = 999., 999.
    best_pred, best_y = None, None

    early_stop, early_stop_max, num_epochs = 0., args.early_stop, args.num_epochs

    for epoch in range(num_epochs):

        epoch_loss = 0.

        with tqdm(total=len(train_loader)) as pbar:

            for batch_idx, batch in enumerate(train_loader):

                batch_X_1, batch_X_2, batch_y = batch

                # Convert numpy arrays to torch tensors

                batch_y = batch_y.to(gpu)
                mask1 = model.create_padding_mask(batch_X_1, 0) ## 항체 길이에 맞는 padding mask 생성
                mask1 = mask1.to(gpu)

                batch_y = batch_y.to(gpu)
                mask2 = model.create_padding_mask(batch_X_2, 0) ## 표적 단백질 길이에 맞는 padding mask 생성
                mask2 = mask2.to(gpu)

                batch_X_1 = batch_X_1.to(gpu)
                batch_X_2 = batch_X_2.to(gpu)

                # Forward Pass
                model.train()
                
                output = model(batch_X_1, batch_X_2, mask1, mask2)
                train_loss = criterion(output.float(), batch_y.unsqueeze(1).float())

                with torch.no_grad():
                    epoch_loss += train_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                pbar.update(1)
            train_loss_arr.append(epoch_loss / len(train_loader))

            # 기존
            if epoch % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    ## 각 dataset에 대한 loss 계산
                    train_loss = evaluate_category(
                        model, train_loader, gpu, criterion)
                    val_loss = evaluate_category(
                        model, valid_loader, gpu, criterion)
                    test_loss = evaluate_category(
                        model, test_loader, gpu, criterion)

                    val_loss_arr.append(val_loss)
                    test_loss_arr.append(test_loss)
                    print("=" * 100)
                    if best_loss > val_loss:
                        best_ACC = val_loss
                        early_stop = 0

                        final_ACC = test_loss
                        print_result_category(epoch, num_epochs, train_loss, val_loss, is_improved=True)

                        if is_save:
                            torch.save(model.state_dict(), model_save_path)

                    else:
                        early_stop += 1 ## early_stop은 제대로 구현 안함
                        print_result_category(epoch, num_epochs, train_loss, val_loss, is_improved=False)
    
    print(final_ACC) ## 최종 test loss 출력


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001) ## learning rate
    parser.add_argument('--reg', type=float, default=0) ## weight decay
    parser.add_argument('--batch_size', type=int, default=512) ## batch_size
    parser.add_argument('--num_epochs', type=int, default=200) ## num_epochs
    parser.add_argument('--early_stop', type=int, default=50) ## early_stop 

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
    model_save_path += 'Transformer_regression'

    if model_save == 0:
        model_save = False
    elif model_save == 1:
        model_save = True
    else:
        print("check model save argument")
        exit()

    # for train
    MAX_LEN = 60
    lr, batch_size, reg = opt.lr, opt.batch_size, opt.reg

    ## dataset
    df = pd.read_csv('CATNAP_data.csv') # dataset file here

    train_data, valid_data, test_data = np.split(df.sample(frac=1).reset_index(drop=True), [int(.8 * len(df)), int(.9 * len(df))]) ## dataset을 8 : 1 : 1 비율로 나눔
    
    ## 각 dataset에 대한 전처리
    train_dataset = FASTADataset(
        sequences1=train_data.FASTA_Ab.to_numpy(),
        sequences2=train_data.FASTA_Virus.to_numpy(),
        labels=train_data.IC50.values,
        auto_padding=MAX_LEN,
        regression=True
    )
    ## 각 dataset에 대한 dataloader 생성
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)

    valid_dataset = FASTADataset(
        sequences1=valid_data.FASTA_Ab.to_numpy(),
        sequences2=valid_data.FASTA_Virus.to_numpy(),
        labels=valid_data.IC50.values,
        auto_padding=MAX_LEN,
        regression=True
    )
    valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True)

    test_dataset = FASTADataset(
        sequences1=test_data.FASTA_Ab.to_numpy(),
        sequences2=test_data.FASTA_Virus.to_numpy(),
        labels=test_data.IC50.values,
        auto_padding=MAX_LEN,
        regression=True
    )
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=True)
    
    model = transformer_regression(26, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=1)
    model.to(gpu)

    criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=lr, weight_decay=reg)
    
    train(opt, gpu, criterion, optimizer, model, train_loader, valid_loader, test_loader, model_save_path, is_save=model_save)
