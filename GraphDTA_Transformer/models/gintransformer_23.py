import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

# GINTransformer model
class GINTransformer_2(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 embed_dim=128, output_dim=128, dropout=0.1, hidden_dim=1024, num_head=4, num_layer=4, drop_prob=0.1, MAX_LEN=1000):

        super(GINTransformer_2, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)
        
        # Transformer on protein sequence
        self.pos_encoder = PositionalEncoding(embed_dim, drop_prob)
        # self.pos_embed = PositionalEncoding(embed_dim, MAX_LEN, device)
        self.embed_dim = embed_dim
        encoder_layers = TransformerEncoderLayer(embed_dim, num_head, hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)
        self.embed = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0) ## word embedding
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, class_nums)
        
        #self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) ## [CLS] token
        #self.pos_embed = nn.Parameter(torch.rand(1, MAX_LEN + 1, embed_dim)) ## position embedding        
        
        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data, mask1):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # Transformer
        # emb_xt = self.embed(target)
        # emb_xt = self.dropout(emb_xt + self.pos_embed(target))

        token_embed = self.embed(target)
        print(token_embed.size())
        pos_embed = self.pos_encoder(target)
        print(pos_embed.size())
        xt = self.dropout(token_embed + pos_embed)
        xt = self.norm(xt)
        # xt = self.embed(target) * math.sqrt(self.embed_dim)
        # cls_token = self.cls_token.expand(xt.shape[0], -1, -1)
        # xt = torch.cat((cls_token, xt), dim=1)
        
        # xt = xt + self.pos_embed
        xt = xt.permute(1, 0, 2)
        xt = self.transformer_encoder(xt, src_key_padding_mask=mask1)
        xt = xt.permute(1, 0, 2)
        xt = xt[:, 0]

        # concat
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
      

