import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool

## 분류 모델
class transformer_classification(nn.Module): 
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5, class_num=0, MAX_LEN=30):
        super(transformer_classification, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        # self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim, padding_idx=0) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, MAX_LEN + 1, embed_dim)) ## position embedding
        self.head = nn.Linear(2*self.embed_dim, class_num) if class_num > 0 else nn.Identity() ## MLP
            
    def forward(self, x1, x2, mask1, mask2):
        ## 항체에 대한 embedding
        x1 = self.embed(x1)
        # x1 = x1 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)
        
        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]

        ## 표적 단백질에 대한 embedding
        x2 = self.embed(x2)
        #x2 = x2 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x2), dim=1)

        x2 = x2 + self.pos_embed
        x2 = x2.permute(1, 0, 2)
        x2 = self.transformer_encoder(x2, src_key_padding_mask=mask2)
        x2 = x2.permute(1, 0, 2)
        x2 = self.norm(x2)
        x2 = x2[:, 0]
        
        cat_x = torch.cat([x1, x2], 1)
        
        # A1 = F.relu(cat_x)
        # A1 = self.dropout(A1)

        x = self.head(cat_x)
        return x 

    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask

## 회귀 모델
class transformer_regression(nn.Module):
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5, class_num=0):
        super(transformer_regression, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        #self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, 61, embed_dim)) ## position embedding
        self.head = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, class_num), nn.ReLU()) if class_num > 0 else nn.Identity()

            
    def forward(self, x1, x2, mask1, mask2):
        ## 항체에 대한 embedding
        x1 = self.embed(x1) * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)

        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]

        ## 표적 단백질에 대한 embedding
        x2 = self.embed(x2) * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x2), dim=1)

        x2 = x2 + self.pos_embed
        x2 = x2.permute(1, 0, 2)
        x2 = self.transformer_encoder(x2, src_key_padding_mask=mask2)
        x2 = x2.permute(1, 0, 2)
        x2 = self.norm(x2)
        x2 = x2[:, 0]
        
        cat_x = torch.cat([x1, x2], 1)
        
        A1 = F.relu(cat_x)
        A1 = self.dropout(A1)

        x = self.head(A1)
        return x 

    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask

# ## 분류 모델
class transformer_Embedding(nn.Module): 
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5):
        super(transformer_Embedding, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        #self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, 61, embed_dim)) ## position embedding
            
    def forward(self, x1, x2, mask1, mask2):
        ## 항체에 대한 embedding
        x1 = self.embed(x1) * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)

        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]

        ## 표적 단백질에 대한 embedding
        x2 = self.embed(x2) * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x2), dim=1)

        x2 = x2 + self.pos_embed
        x2 = x2.permute(1, 0, 2)
        x2 = self.transformer_encoder(x2, src_key_padding_mask=mask2)
        x2 = x2.permute(1, 0, 2)
        x2 = self.norm(x2)
        x2 = x2[:, 0]
        
        cat_x = torch.cat([x1, x2], 1)
        
        # A1 = F.relu(cat_x)
        # A1 = self.dropout(A1)
        return cat_x

class GIN(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                Linear(dim, dim), ReLU())
        )

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)

        return x

class Ensemble(nn.Module):
    def __init__(self, in_channels, dim, out_channels, task):
        super(Ensemble, self).__init__()
        self.transformer = transformer_Embedding(26, embed_dim=128, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1)
        
        self.gin1 = GIN(in_channels, dim, out_channels)
        self.gin2 = GIN(in_channels, dim, out_channels)
        
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(nn.Linear(2*128 + 2*dim, dim+128), nn.ReLU(), nn.Linear(dim+128, 1), nn.ReLU())
        self.fc = nn.Linear(2*128 + 2*dim, 1)
        
        self.task = task
        
    def forward(self, seq_ab, seq_vir,mask1, mask2, x_ab, edge_index_ab, batch_ab, x_vir, edge_index_vir, batch_vir):
        transformer_emb = self.transformer(seq_ab, seq_vir, mask1, mask2)
        gin_emb1 = self.gin1(x_ab, edge_index_ab, batch_ab)
        gin_emb2 = self.gin2(x_vir, edge_index_vir, batch_vir)
        emb = torch.cat((transformer_emb, gin_emb1, gin_emb2), 1)

        if self.task == "classification":
            A1 = F.relu(emb)
            A1 = self.dropout(A1)
            res = self.fc(A1)
            res = torch.sigmoid(self.fc(emb))
        elif self.task == "regression":
            res = self.fc(emb)
        else:
            raise ValueError

        return res
    
    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask
      
## 분류 모델
class transformer_classification_one(nn.Module): 
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5, class_num=0):
        super(transformer_classification_one, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        #self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, 31, embed_dim)) ## position embedding
        self.head = nn.Linear(self.embed_dim, class_num) if class_num > 0 else nn.Identity() ## MLP
            
    def forward(self, x1, mask1):
        ## 항체에 대한 embedding
        x1 = self.embed(x1) * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)

        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]
        
        A1 = F.relu(x1)
        A1 = self.dropout(A1)

        x = self.head(A1)
        return x 

    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask
      
def aa_features():
    # Meiler's features
    prop1 = [[1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
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
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]]
    return np.array(prop1)      
  
def seq_to_one_hot(res_seq_one):
    ints = one_to_number(res_seq_one)
    feats = aa_features()[ints]
    onehot = to_categorical(ints, num_classes=len(aa_s))
    return np.concatenate((onehot, feats), axis=1)
      
## 분류 모델
class transformer_classification_added(nn.Module): 
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5, class_num=0, MAX_LEN=30):
        super(transformer_classification_added, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        #self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
  
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim - 7, padding_idx=0) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, MAX_LEN + 1, embed_dim)) ## position embedding
        self.head = nn.Linear(2*self.embed_dim, class_num) if class_num > 0 else nn.Identity() ## MLP
            
    def forward(self, x1, x2, feats1, feats2, mask1, mask2):
        ## 항체에 대한 embedding
        x1 = torch.cat((self.embed(x1), feats1), dim=2) 
        # x1 = x1 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)

        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]

        ## 표적 단백질에 대한 embedding
        x2 = torch.cat((self.embed(x2), feats2), dim=2) 
        # x2 = x2 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x2), dim=1)

        x2 = x2 + self.pos_embed
        x2 = x2.permute(1, 0, 2)
        x2 = self.transformer_encoder(x2, src_key_padding_mask=mask2)
        x2 = x2.permute(1, 0, 2)
        x2 = self.norm(x2)
        x2 = x2[:, 0]
        
        cat_x = torch.cat([x1, x2], 1)
        x = torch.sigmoid(self.head(cat_x))
        # A1 = F.relu(cat_x)
        # A1 = self.dropout(A1)
        
        # x = self.head(A1)
        
        # x = self.head(cat_x)
        return x 

    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask

class transformer_classification_one_added(nn.Module): 
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5, class_num=0, MAX_LEN=30):
        super(transformer_classification_one_added, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        #self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim - 7, padding_idx=0) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, MAX_LEN + 1, embed_dim)) ## position embedding
        self.head = nn.Linear(self.embed_dim, class_num) if class_num > 0 else nn.Identity() ## MLP
            
    def forward(self, x1, feats1, mask1):
        ## 항체에 대한 embedding
        x1 = torch.cat((self.embed(x1), feats1), dim=2) 
        # x1 = x1 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)

        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]
        
        A1 = F.relu(x1)
        A1 = self.dropout(A1)

        x = self.head(A1)
        return x 

    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask
      
## 분류 모델
class transformer_rank_added(nn.Module): 
    def __init__(self, token_num, embed_dim=512, hidden_dim=2048, num_head=8, num_layer=6, drop_prob=0.5, class_num=0, MAX_LEN=30):
        super(transformer_rank_added, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.embed_dim = embed_dim
        self.token_num = token_num
        #self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
  
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(self.token_num, self.embed_dim - 7, padding_idx=0) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        self.pos_embed = nn.Parameter(torch.rand(1, MAX_LEN + 1, embed_dim)) ## position embedding
        self.head = nn.Linear(2*self.embed_dim, class_num) if class_num > 0 else nn.Identity() ## MLP
            
    def forward(self, x1, x2, feats1, feats2, mask1, mask2):
        ## 항체에 대한 embedding
        x1 = torch.cat((self.embed(x1), feats1), dim=2) 
        # x1 = x1 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_token, x1), dim=1)

        x1 = x1 + self.pos_embed
        x1 = x1.permute(1, 0, 2)
        x1 = self.transformer_encoder(x1, src_key_padding_mask=mask1)
        x1 = x1.permute(1, 0, 2)
        x1 = self.norm(x1)
        x1 = x1[:, 0]

        ## 표적 단백질에 대한 embedding
        x2 = torch.cat((self.embed(x2), feats2), dim=2) 
        # x2 = x2 * math.sqrt(self.embed_dim)
        cls_token = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_token, x2), dim=1)

        x2 = x2 + self.pos_embed
        x2 = x2.permute(1, 0, 2)
        x2 = self.transformer_encoder(x2, src_key_padding_mask=mask2)
        x2 = x2.permute(1, 0, 2)
        x2 = self.norm(x2)
        x2 = x2[:, 0]
        
        cat_x = torch.cat([x1, x2], 1)
        # x = torch.sigmoid(self.head(cat_x))
        
        A1 = F.relu(cat_x)
        A1 = self.dropout(A1)
        
        x = self.head(A1)
        
        # x = self.head(cat_x)
        return x 

    def initHidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))
    
    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx):
        mask = (seq == pad_idx)
        zeros = torch.zeros(seq.shape[0], 1).bool()
        mask = torch.cat((zeros, mask), dim=1)
        return mask