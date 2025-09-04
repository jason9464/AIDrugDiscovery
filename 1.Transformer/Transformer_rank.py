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
from model import transformer_rank_added

def load_trained_model(model_save_path, gpu):

    # model
    model = transformer_rank_added(21, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=3, MAX_LEN=30)
    
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
            
            a, b, c = torch.FloatTensor([1] * X_1.size(0)), torch.FloatTensor([2] * X_1.size(0)), torch.FloatTensor([3] * X_1.size(0))
            value = torch.stack([a, b, c]).T
            value = value.to(gpu)

            output = model(X_1, X_2, feats1, feats2, mask1, mask2)
            A = F.softmax(output, dim=1)
            A = (A * value).sum(1).squeeze()
                        
            output_list.extend(A-1.)
            y_list.extend(y)

            antibody_list.extend(antibody)
            antigen_list.extend(antigen)
            
            total_loss += torch.sum(criterion(A, y+1.))

        output_list = torch.stack(output_list)
        output_list = torch.squeeze(output_list)
        y_list = torch.stack(y_list)
        y_list = torch.squeeze(y_list)

        output_list = torch.clamp(output_list, 0, 2).round()
        
        if test == True:
            pred_dict = {"CDRH3": antibody_list, "CDRL3": antigen_list, "pred_label": [int(i) for i in output_list], "Class": [int(i) for i in y_list]}
            pd.DataFrame(pred_dict).to_csv("rank_testdataset_result.csv")

        ACC = float((y_list == output_list).sum()) / y_list.size(0)

    return total_loss / len(test_loader), ACC

def print_result_category(epoch, num_epochs, train_loss, ACC_train, ACC_test, is_improved=False): ## 결과 출력 함수
    if is_improved:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train ACC: {:.4f}, Test ACC: {:.4f} *'.format(
            epoch, num_epochs, train_loss, ACC_train, ACC_test))
    else:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train ACC: {:.4f}, Test ACC: {:.4f}'.format(
            epoch, num_epochs, train_loss, ACC_train, ACC_test))

def train(args, gpu, criterion1, criterion2, optimizer, model, train_loader, test_loader, is_save=False): ## 모델 훈련
    train_loss_arr = []
    test_loss_arr = []

    best_ACC, final_ACC = -999., -999.
    best_pred, best_y = None, None

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
                
                a, b, c = torch.FloatTensor([1] * batch_X_1.size(0)), torch.FloatTensor([2] * batch_X_1.size(0)), torch.FloatTensor([3] * batch_X_1.size(0))
                value = torch.stack([a, b, c]).T
                value = value.to(gpu)
                    
                # Forward Pass
                model.train()
                
                output = model(batch_X_1, batch_X_2, batch_feats1, batch_feats2, mask1, mask2)
                A = F.softmax(output, dim=1)
                A = (A * value).sum(1).squeeze()

                train_loss = torch.sum(criterion1(A, batch_y + 1.) + criterion2(output, batch_y))

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
                        model, train_loader, gpu, criterion1)
                    test_loss, ACC_test = evaluate_category(
                        model, test_loader, gpu, criterion1, False)

                    test_loss_arr.append(test_loss)
                    print("=" * 100)
                    if best_ACC < ACC_test:
                        best_ACC = ACC_test
                        early_stop = 0

                        final_ACC = ACC_test
                        print_result_category(epoch, num_epochs, train_loss, ACC_train, ACC_test, is_improved=True)

                        if is_save:
                            torch.save(model.state_dict(), model_save_path)

                    else:
                        early_stop += 1 ## early_stop은 제대로 구현 안함
                        print_result_category(epoch, num_epochs, train_loss, ACC_train, ACC_test, is_improved=False)
    
    print(final_ACC) ## 최종 test loss 출력


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0001) ## learning rate
    parser.add_argument('--reg', type=float, default=0) ## weight decay
    parser.add_argument('--batch_size', type=int, default=512) ## batch_size
    parser.add_argument('--num_epochs', type=int, default=200) ## num_epochs
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
    model_save_path += 'Transformer_rank'

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
    df = pd.read_csv('SARS-CoV2_Omicron-BA1_IC50_20230110.csv') # dataset file here

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=1004, stratify=df.label)
    # train_data, valid_data, test_data = np.split(df.sample(frac=1).reset_index(drop=True), [int(.8 * len(df)), int(.9 * len(df))]) ## dataset을 8 : 1 : 1 비율로 나눔

    ## 각 dataset에 대한 전처리
    train_dataset = FASTADataset(
        sequences1=train_data.CDRH3.to_numpy(),
        sequences2=train_data.CDRL3.to_numpy(),
        labels=train_data.label.to_numpy(),
        auto_padding=MAX_LEN,
        regression=False
    )
    ## 각 dataset에 대한 dataloader 생성
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = FASTADataset(
        sequences1=test_data.CDRH3.to_numpy(),
        sequences2=test_data.CDRL3.to_numpy(),
        labels=test_data.label.to_numpy(),
        auto_padding=MAX_LEN,
        regression=False
    )
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=True)
    
    model = transformer_rank_added(21, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, class_num=3, MAX_LEN=MAX_LEN)
    model.to(gpu)
    
    criterion1 = nn.MSELoss(reduce=False)
    criterion2 = torch.nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=lr, weight_decay=reg)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(opt, gpu, criterion1, criterion2, optimizer, model, train_loader, test_loader, is_save=model_save)
    
    model1 = load_trained_model(model_save_path, gpu)
    
    test_loss, ACC_test = evaluate_category(model1, test_loader, gpu, criterion1, True)
    
    print('Final Test ACC: {:.4f}'.format(ACC_test))
