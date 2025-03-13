import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from PIL import Image

from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from Utils import create_attention_mask_from_input_mask, create_data_from_input_mask
from Models import TransformerEncoder


class FTransformer(nn.Module):
    def __init__(self, feature_size: int=25, patch_size: int=5, num_layers: int=6, num_heads: int=4,
        d_model: int=64, hidden: int=256, dropout: float=0.1, attention_dropout: float=0.0, 
        n_classes: int=1, **kwargs):
        super(FTransformer, self).__init__()
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.hidden = hidden
        self.dropout = dropout
        self.attn_dropout = attention_dropout
        self.n_classes = n_classes
        self.n_seq = feature_size // patch_size

        self.reproj = nn.Linear(patch_size, d_model)
        self.encoder = TransformerEncoder(seq_length=self.n_seq+1, num_layers=num_layers, num_heads=num_heads,
                                          d_model=d_model,hidden=hidden,dropout=dropout,attention_dropout=self.attn_dropout)
        self.d0 = nn.Linear(d_model, n_classes)
        self.d1 = nn.Linear(d_model, n_classes)
        # self.pooler = nn.Sequential(
        #     nn.Linear(d_model, hidden),
        #     nn.LayerNorm(hidden),
        #     nn.Linear(hidden, 2)
        # )
        # self.reg = nn.Sequential(
        #     nn.Linear(d_model, hidden),
        #     nn.LayerNorm(hidden),
        #     # nn.GELU(),
        #     nn.Linear(hidden, patch_size)
        # )
        self.reg = nn.Linear(d_model, patch_size)
        self.pooler = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_type_table = nn.Parameter(torch.empty(4, d_model).normal_(std=0.02))
        self.seg_type_table = nn.Parameter(torch.empty(2, d_model).normal_(std=0.02))
        self.pos_embedding = nn.Parameter(torch.empty(1, 2*self.n_seq + 2, d_model).normal_(std=0.02))
        self.seq_length = self.n_seq + 1
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        p = self.patch_size
        x = rearrange(x, 'B (N P) -> B N P', P=p)
        x = self.reproj(x)
        self.token_type_ids = torch.zeros(B, self.n_seq).to(torch.int64)
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, token_type=True, position_emb=True):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        B, N, P = x.shape

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        if token_type:
            token_type_ids = torch.concatenate([torch.ones(B, 1).to(torch.int64)+1, self.token_type_ids], dim=1).reshape(B*(N+1))
            one_hot_ids = torch.nn.functional.one_hot(token_type_ids, num_classes=4)
            token_type_embeddings = torch.matmul(one_hot_ids.float(), self.token_type_table).reshape(B, N+1, self.d_model)
            x += token_type_embeddings
        if position_emb:
            position_embedding = self.pos_embedding[:,:N+1,:]
            x += position_embedding

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        pred0 = self.d0(x)
        pred1 = self.d1(x)
        yf = torch.where(t>0, pred1, pred0).to(float)
        ycf = torch.where(t>0, pred0, pred1).to(float)
        return yf, ycf
    
    def pretrain(self, x_from: torch.Tensor, x_to: torch.Tensor, seg_type:bool=True, token_type: bool=True, position_emb: bool=True, device='cpu'):
        input_from = self._process_input(x_from)
        input_to = self._process_input(x_to)
        B, N, P = input_from.shape
        B_, N_, P_ = input_to.shape

        batch_class_token = self.class_token.expand(B, -1, -1)
        sep_token = self.sep_token.expand(B, -1, -1)
        x = torch.cat([batch_class_token, input_from, sep_token, input_to], dim=1)
        if seg_type:
            seg_type_ids = torch.concatenate([torch.zeros(B, N+2).to(torch.int64), 
                                                torch.ones(B, N_).to(torch.int64)], dim=1).reshape(B*(N+2+N_))
            one_hot_ids = torch.nn.functional.one_hot(seg_type_ids, num_classes=2)
            seg_type_embeddings = torch.matmul(one_hot_ids.float(), self.seg_type_table).reshape(B, N+2+N_, self.d_model)
            x += seg_type_embeddings
        if token_type:
            token_type_ids = torch.concatenate([torch.ones(B, 1).to(torch.int64)+1, self.token_type_ids,
                                                torch.ones(B, 1).to(torch.int64)+2, self.token_type_ids], dim=1).reshape(B*(N+2+N_))
            # token_type_table = nn.Parameter(torch.empty(4, self.d_model).normal_(std=0.02))
            one_hot_ids = torch.nn.functional.one_hot(token_type_ids, num_classes=4)
            token_type_embeddings = torch.matmul(one_hot_ids.float(), self.token_type_table).reshape(B, N+2+N_, self.d_model)
            x += token_type_embeddings

        input_mask_from = torch.empty((B, N), dtype=torch.int64).bernoulli_(p=0.8)
        input_mask_to = torch.empty((B, N_), dtype=torch.int64).bernoulli_(p=0.8)
        input_mask = torch.cat([torch.ones(B, 1).to(torch.int64), input_mask_from, 
                                torch.ones(B, 1).to(torch.int64), input_mask_to], dim=1)
        inputs = create_data_from_input_mask(x.detach().cpu(), input_mask)
        attn_mask = create_attention_mask_from_input_mask((B, 2+N+N_), input_mask)
        attn_mask = attn_mask.repeat(1, 1, self.num_heads).reshape(B*self.num_heads, 2+N+N_, 2+N+N_).to(device)
        # attn_mask = -attn_mask + 1
        # attn_mask = attn_mask.to(torch.bool)
        attn_mask = torch.where(attn_mask>0, 0., -float('inf'))
        torch._assert(inputs.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {inputs.shape}")
        if position_emb:
            inputs += self.pos_embedding

        for i in range(self.num_layers):
            self.encoder.layers[i].attn_mask = attn_mask

        x = self.encoder(inputs)

        output_mask = -input_mask+1
        mask_cont = torch.cat([output_mask[:,1:N+1], output_mask[:,N+2:]], dim=1).reshape(B*(N + N_))
        mask_cont_ = torch.nonzero(mask_cont)
        reps_cont = torch.cat([x[:,1:N+1,:], x[:,N+2:,:]], dim=1).reshape(B*(N + N_), self.d_model)
        reps_cont_ = reps_cont[mask_cont_.squeeze()]
        label_cont = torch.cat([x_from, x_to], dim=1).reshape(B*(N + N_), self.patch_size)
        label_cont_ = label_cont[mask_cont_.squeeze()]
        preds_conts = self.reg(reps_cont_)
        masked_lm_loss = torch.nn.functional.mse_loss(preds_conts, label_cont_)

        x = x[:, 0]
        logits = self.pooler(x)
        return logits, masked_lm_loss