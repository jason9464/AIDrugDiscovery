import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gintransformer import GINTransformer
from utils import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        mask1 = create_padding_mask(data.target, 0)
        mask1 = mask1.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, mask1)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss.item()

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mask1 = create_padding_mask(data.target, 0)
            mask1 = mask1.to(device)
            data = data.to(device)
            output = model(data, mask1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = [['BindingDB'][int(sys.argv[1])]] 
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, GINTransformer][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 700

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file = 'data/processed/' + dataset + '_train_30000.pt'
    if (not os.path.isfile(processed_data_file)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        dataset_full = TestbedDataset(root='data', dataset=dataset+'_train_30000')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(dataset_full, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        # training the model
        device = torch.device(cuda_name)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_loss_arr = []
        model_file_name = 'model_' + model_st + '_' + dataset + str(LR) + '_' + str(NUM_EPOCHS) + '_30000_log6' + '.model'
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, device, train_loader, optimizer, epoch+1)
            train_loss_arr.append(train_loss)
            
            torch.save(model.state_dict(), model_file_name)
