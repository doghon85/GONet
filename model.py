import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LocalLinear(nn.Module):
    def __init__(self, in_features, local_features, kernel_size, stride=1, bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - 1

        fold_num = (in_features+self.padding -self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num, kernel_size, local_features)) # (44314, 5, 1)
        self.bias = nn.Parameter(torch.randn(fold_num, local_features)) if bias else None # (44314, 1)

        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x:torch.Tensor): # (16, 44314)
        x = F.pad(x,[0, self.padding],value=0) # (16, 44318) / pad(x, [3,4]) 이면 3 + x + 4 로 padding
        x = x.unfold(-1,size=self.kernel_size,step=self.stride) # (16, 44314, 5) unfold(axis, num_of_components, stride)
        x = torch.matmul(x.unsqueeze(2), self.weight).squeeze(2)+self.bias # (44314, 1)
        return x.squeeze(2)

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PE(nn.Module):
    def __init__(self):
        super().__init__()
        self.PE = nn.Parameter()
    def forward(self, x):
        return x + self.PE

class GONet(nn.Module):
    def __init__(self, ymean, num_snp, device):
        super(GONet, self).__init__()

        # set mean
        self.mean = ymean
        self.device = device
        
        # set LCL
        self.encoder = nn.Sequential(
            LocalLinear(num_snp, 1, kernel_size=5, stride=1),
            nn.LayerNorm(num_snp),
            nn.GELU(),
            LocalLinear(num_snp, 1, kernel_size=3, stride=1),
            nn.LayerNorm(num_snp),
            nn.GELU()
        )

        self.com_encoder = nn.Sequential(
            nn.Linear(num_snp, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.PE = nn.Parameter(torch.rand(512))
        self.query = nn.Linear(128, 128)
        self.key_value = nn.Linear(128, 128)
        self.inter_individual_cross_attn = nn.MultiheadAttention(128, 4, batch_first=True)

        self.MLP = MLP(input_dim=1024, hidden_dim=256, output_dim=128, num_layers=2)
        self.order = nn.Linear(128, 3)
        self.bias = nn.Linear(1024, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, X, train=False):
        X = self.encoder(X) + X
        com = self.com_encoder(X)
        order, v1, v2 = None, None, None
        if train:
            temp1, temp2 = com.split(com.shape[0]//2, dim=0)
            temp1_, temp2_ = temp1.reshape(temp1.shape[0], 4, -1), temp2.reshape(temp2.shape[0], 4, -1)

            pos_temp1, pos_temp2 = temp1.reshape(temp1.shape[0], -1) + self.PE, temp2.reshape(temp2.shape[0], -1) + self.PE 
            pos_temp1_, pos_temp2_ = pos_temp1.reshape(pos_temp1.shape[0], 4, -1), pos_temp2.reshape(pos_temp2.shape[0], 4, -1) 

            q1 = self.query(pos_temp1_)
            v1 = self.key_value(temp1_)

            q2 = self.query(pos_temp2_) 
            v2 = self.key_value(temp2_)

            attention_score1 = self.inter_individual_cross_attn(query=q1, key=q2, value=v2)[0] 
            attention_score1 = attention_score1.reshape(q1.shape[0], -1) + temp1 

            attention_score2 = self.inter_individual_cross_attn(query=q2, key=q1, value=v1)[0] 
            attention_score2 = attention_score2.reshape(q1.shape[0], -1) + temp2  

            attention_score = torch.concat([attention_score1, attention_score2], dim=1)
            attention_score = self.MLP(attention_score)

            order = self.order(attention_score)

        else:
            v1, v2 = com.split(com.shape[0]//2, dim=0)
            v1, v2 = v1.reshape(v1.shape[0], 4, -1), v2.reshape(v2.shape[0], 4, -1)
            
            v1 = self.key_value(v1)
            v2 = self.key_value(v2)
        
        value = torch.concat([v1.reshape(v1.shape[0], -1), v2.reshape(v2.shape[0], -1)], dim=0) 
        b = self.bias(torch.concat([com, value], dim=1)).squeeze(1) 

        return self.mean + b, order


