import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from nystrom_lib.nystrom_attention import NystromAttention
def sanity_check(tensor):
	min_value = tensor.min().abs()
	order_of_magnitude = torch.log10(min_value)
	if order_of_magnitude < 0:
		mul_val = order_of_magnitude.abs().item() 
		scientific_notation = float(f'1e+{int(mul_val):d}')
	else:
		mul_val = order_of_magnitude.item()
		scientific_notation = float(f'1e-{int(mul_val):d}')
	return scientific_notation
  
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=256):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse
            residual = True,         # whether to do an extra residual with the value or not
            dropout=0.1
        )


    def forward(self, x):
        out, att,padding, n = self.attn(self.norm(x)) 
        x = x + out
        return x, att, padding, n


class PPEG(nn.Module):
    #def __init__(self, dim=512):
    def __init__(self, dim=256):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=256) 
        self._fc1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU()) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256)) 
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=256) 
        self.layer2 = TransLayer(dim=256) 
        self.attention = nn.Sequential(nn.Linear(256,256), nn.Tanh(), nn.Linear(256,1)) 
        self._fc2 = nn.Linear(256, self.n_classes) 


    def relocate(self):
        #check for GPU
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_layer = self.pos_layer.to(device)
        self._fc1 = self._fc1.to(device)
        self.cls_token = self.cls_token.to(device)
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)
        #self.norm = self.norm.to(device)
        self.attention = self.attention.to(device)
        self._fc2 = self._fc2.to(device)
    
    def forward(self, h, label=None, mn_M=None):
        device = h.device
        h = h.float()
        h = self._fc1(h) 
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) 
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        
        h, _, _, _ = self.layer1(h) #Translayer 1
        h = self.pos_layer(h, _H, _W) #PPEG
        h, _, _, _ = self.layer2(h) #Translayer 2
        a = self.attention(h)  #Attention pooling
        a = a.transpose(1, 2) 
        a = F.softmax(a, dim=2) 
        att = a.squeeze()
        att = att[1:-add_length]
        h = torch.bmm(a, h).squeeze(1) 
        min_val = att.min()
        max_val = att.max()
        mul_with = sanity_check(att)
        att = mul_with * att
        min_val = att.min()
        max_val = att.max()
        att = (att - min_val) / (max_val - min_val)
        min_val = att.min()
        max_val = att.max()

        logits = self._fc2(h) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        if mn_M is None: mn_M = att
        return logits, Y_prob, Y_hat, att, mn_M

