import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import numpy as np

aa_s = "CSTPAGNDEQHRKMILVFYW" # X for unknown
  
def aa_features():
    # Meiler's features
    prop1 = [[0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00],
             [1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42]]
    return np.array(prop1)      
  
def seq_to_one_hot(res_seq_one):
    feats = aa_features()[ints]
    return feats
  
def one_to_number(r):
    try:
        return aa_s.index(r) + 1
    except:
        return 0
  
class FASTADataset(Dataset):
    def __init__(self, sequences1, sequences2, labels, auto_padding=0, regression=False, is_test_data=False):
        self.sequences1 = sequences1
        self.sequences2 = sequences2
        self.labels = labels
        self.auto_padding = auto_padding
        self.regression = regression
        self.is_test_data = is_test_data

    def __len__(self):
        return len(self.sequences1)

    def __getitem__(self, idx):
        ## 항체
        sequence1 = ' '.join(str(self.sequences1[idx])) 
        tokens1 = sequence1.split() ## 토큰화
        x1 = [one_to_number(token) for token in tokens1] ## 정수 인코딩
            
        ## zero padding
        if self.auto_padding > 0: 
            if len(x1) < self.auto_padding:
                x1 = x1 + [0]*(self.auto_padding - len(x1))
            elif len(x1) > self.auto_padding:
                x1 = x1[:self.auto_padding]
                
        feats1 = aa_features()[x1]

        ## 표적 단백질
        sequence2 = ' '.join(str(self.sequences2[idx]))
        tokens2 = sequence2.split() ## 토큰화
        x2 = [one_to_number(token) for token in tokens2] ## 정수 인코딩

        ## zero padding
        if self.auto_padding > 0:
            if len(x2) < self.auto_padding:
                x2 = x2 + [0]*(self.auto_padding - len(x2))
            elif len(x2) > self.auto_padding:
                x2 = x2[:self.auto_padding]

        feats2 = aa_features()[x2]

        if self.is_test_data:
            return  torch.LongTensor(x1), torch.LongTensor(x2), torch.tensor(feats1, dtype = torch.float32), torch.tensor(feats2, dtype = torch.float32)
          
        elif self.regression:
            # label = self.labels[idx]
            label = -math.log10(self.labels[idx] * pow(10, -6))

        if not self.regression: ## 분류
            return  torch.LongTensor(x1), torch.LongTensor(x2), torch.tensor(feats1, dtype = torch.float32), torch.tensor(feats2, dtype = torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.sequences1[idx], self.sequences2[idx]

        else: ## 회귀
            return  torch.LongTensor(x1), torch.LongTensor(x2), torch.tensor(label), self.sequences1[idx], self.sequences2[idx]
          
          
class FASTADataset_one(Dataset):
    def __init__(self, sequences1, labels, auto_padding=0, regression=False, is_test_data=False):
        self.sequences1 = sequences1
        self.labels = labels
        self.auto_padding = auto_padding
        self.regression = regression
        self.is_test_data = is_test_data

    def __len__(self):
        return len(self.sequences1)

    def __getitem__(self, idx):
        ## 항체
        sequence1 = ' '.join(str(self.sequences1[idx])) 
        tokens1 = sequence1.split() ## 토큰화
        x1 = [one_to_number(token) for token in tokens1] ## 정수 인코딩

        ## zero padding
        if self.auto_padding > 0: 
            if len(x1) < self.auto_padding:
                x1 = x1 + [0]*(self.auto_padding - len(x1))
            elif len(x1) > self.auto_padding:
                x1 = x1[:self.auto_padding]
                
        feats1 = aa_features()[x1]
        
        if self.is_test_data:
            return  torch.LongTensor(x1), torch.LongTensor(x2)
        elif self.regression:
            # label = self.labels[idx]
            label = -math.log10(self.labels[idx] * pow(10, -6))

        if not self.regression: ## 분류
            return  torch.LongTensor(x1), torch.tensor(feats1, dtype = torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.sequences1[idx]

        else: ## 회귀
            return  torch.LongTensor(x1), torch.LongTensor(x2), torch.tensor(label), self.sequences1[idx], self.sequences2[idx]
          