import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset

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
        x1 = [ord(token)-64 for token in tokens1] ## 정수 인코딩

        ## zero padding
        if self.auto_padding > 0: 
            if len(x1) < self.auto_padding:
                x1 = x1 + [0]*(self.auto_padding - len(x1))
            elif len(x1) > self.auto_padding:
                x1 = x1[:self.auto_padding]
        ## 표적 단백질
        sequence2 = ' '.join(str(self.sequences2[idx]))
        # sequence2 = ' '.join('RVVPSGDVVRFPNITNLCPFGEVFNATKFPSVYAWERKKISNCVADYSVLYNSTFFSTFKCYGVSATKLNDLCFSNVYADSFVVKGDDVRQIAPGQTGVIADYNYKLPDDFMGCVLAWNTRNIDATSTGNYNYKYRLFRKSNLKPFERDISTEIYQAGSTPCNGVAGFNCYFPLRSYSFRPTYGVGHQPYRVVVLSFELLNAPATVCGPKLSTDLIK')
        tokens2 = sequence2.split() ## 토큰화
        x2 = [ord(token)-64 for token in tokens2] ## 정수 인코딩

        ## zero padding
        if self.auto_padding > 0:
            if len(x2) < self.auto_padding:
                x2 = x2 + [0]*(self.auto_padding - len(x2))
            elif len(x2) > self.auto_padding:
                x2 = x2[:self.auto_padding]
        
        if self.is_test_data:
            return  torch.LongTensor(x1), torch.LongTensor(x2)
        elif self.regression:
            # label = self.labels[idx]
            label = -math.log10(self.labels[idx] * pow(10, -6))

        if not self.regression: ## 분류
            return  torch.LongTensor(x1), torch.LongTensor(x2), torch.tensor(self.labels[idx], dtype=torch.long), self.sequences1[idx], self.sequences2[idx]

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
        x1 = [ord(token)-64 for token in tokens1] ## 정수 인코딩

        ## zero padding
        if self.auto_padding > 0: 
            if len(x1) < self.auto_padding:
                x1 = x1 + [0]*(self.auto_padding - len(x1))
            elif len(x1) > self.auto_padding:
                x1 = x1[:self.auto_padding]
        
        if self.is_test_data:
            return  torch.LongTensor(x1), torch.LongTensor(x2)
        elif self.regression:
            # label = self.labels[idx]
            label = -math.log10(self.labels[idx] * pow(10, -6))

        if not self.regression: ## 분류
            return  torch.LongTensor(x1), torch.tensor(self.labels[idx], dtype=torch.long), self.sequences1[idx]

        else: ## 회귀
            return  torch.LongTensor(x1), torch.LongTensor(x2), torch.tensor(label), self.sequences1[idx], self.sequences2[idx]
          