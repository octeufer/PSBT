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

class ViT_custom(nn.Module):
    def __init__(self, dropout=0.1, n_classes=100, d_model=768, hidden=3072, **kwargs):
        super(ViT_custom, self).__init__()
        self.dropout - dropout
        self.n_classes = n_classes
        self.d_model = d_model
        # pre-trained parameter 'ViT_B_16_Weights,IMAGENET1K_V1'
        self.encoder_model = torchvision.models.vit_b_16(weights='ViT_B_16_Weights,IMAGENET1K_V1', dropout=dropout)
        self.cls_heads = nn.Linear(d_model, n_classes)
        self.bit_heads = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.encoder_model._process_input(x)
        n = x.size(0)
        batch_class_token = self.encoder_model.class_token.expand(n,-1,-1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder_model.encoder(x)
        x = x[:,0]
        x = self.cls_heads(x)
        return x
    
    def masked_prediction(self, x, input_mask=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        x = self.encoder_model._process_input(x)
        B, N, D = x.size()
        if not input_mask:
            input_mask = torch.empty((B, N), dtype=torch.int32).bernoulli_(p=0.5)
        output_mask = input_mask.to(bool)*-1+1
        inputs = create_data_from_input_mask(x.detach().cpu(), input_mask)
        batch_class_token = self.encoder_model.class_token.expand(B, -1, -1)
        x = torch.cat([batch_class_token, inputs.to(device)], dim=1)
        input_mask = torch.cat

class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model=48, nhead=8, dim_feedforward=192, dropout=0.1, attention_dropout=0.0,
               activation="gelu", normalize_before=True, attn_mask=None):
    super().__init__()
    self.multihead_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout, batch_first=True)
    # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(
       nn.Linear(d_model, dim_feedforward),
       nn.GELU(),
       nn.Dropout(dropout),
       nn.Linear(dim_feedforward, d_model),
       nn.Dropout(dropout)
    )
    self.attn_mask = attn_mask

#   def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#     return tensor if pos is None else tensor + pos

  def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    # torch._assert(self.attn_mask is not None, f"Expected real attention mask got {self.attn_mask}")
    x = self.norm1(input)
    x, _ = self.multihead_self_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)
    x = self.dropout(x)
    x = x + input

    y = self.norm2(x)
    y = self.mlp(y)
    return x + y

class TransformerEncoder(nn.Module):
    def __init__(self, seq_length: int=8, num_layers: int=6, num_heads: int=8,
        d_model: int=48, hidden: int=192, dropout: float=0.1, attention_dropout: float=0.0):
        super().__init__()
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, d_model).normal_(std=0.02))  # from BERT
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=hidden,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # input = input + self.pos_embedding
        return self.ln(self.layers((self.dropout(input))))


class FTransformer(nn.Module):
    def __init__(self, feature_size: int=25, patch_size: int=5, num_layers: int=8, num_heads: int=8,
        d_model: int=128, hidden: int=512, dropout: float=0.1, attention_dropout: float=0.0, 
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

        self.conte_reproj = nn.Linear(3, d_model)
        self.bin_reproj = nn.Linear(patch_size, d_model)
        self.rest_reproj = nn.Linear(4, d_model)
        self.encoder = TransformerEncoder(seq_length=self.n_seq+1, num_layers=num_layers, num_heads=num_heads,
                                          d_model=d_model,hidden=hidden,dropout=dropout,attention_dropout=self.attn_dropout)
        self.d0 = nn.Linear(d_model, n_classes)
        self.d1 = nn.Linear(d_model, n_classes)
        self.pooler = nn.Linear(d_model, n_classes)
        self.reg = nn.Linear(d_model, 3)
        self.cls_bins = nn.Linear(d_model, 5)
        self.cls_bin = nn.Linear(d_model, 4)

        self.norm = nn.LayerNorm(d_model)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.token_type_table = nn.Parameter(torch.empty(4, d_model).normal_(std=0.02))
        self.seg_type_table = nn.Parameter(torch.empty(2, d_model).normal_(std=0.02))
        self.pos_embedding = nn.Parameter(torch.empty(1, 2*self.n_seq + 4, d_model).normal_(std=0.02))
        self.seq_length = self.n_seq + 1
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        p = self.patch_size
        x_cont = rearrange(x[:,:6], 'B (N P) -> B N P', P=3)
        x_bin = rearrange(x[:,6:21], 'B (N P) -> B N P', P=p)
        x_rest = x[:,21:].reshape(B, 1, 4)
        # x = rearrange(x, 'B (N P) -> B N P', P=p)
        # x = self.reproj(x)
        x_cont_ = self.conte_reproj(x_cont)
        x_bin_ = self.bin_reproj(x_bin)
        x_rest_ = self.rest_reproj(x_rest)
        x = torch.cat([x_cont_, x_bin_, x_rest_], dim=1)
        self.n_seq = x.shape[1]
        self.token_type_ids = torch.cat([torch.zeros(B, x_cont.shape[1]).to(torch.int64),
                                                torch.ones(B, x_bin.shape[1]).to(torch.int64),
                                                torch.ones(B, 1).to(torch.int64)], dim=1)
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
            one_hot_ids = one_hot_ids.to('mps')
            seg_type_embeddings = torch.matmul(one_hot_ids.float(), self.seg_type_table).reshape(B, N+2+N_, self.d_model)
            x += seg_type_embeddings
        if token_type:
            token_type_ids = torch.concatenate([torch.ones(B, 1).to(torch.int64)+1, self.token_type_ids,
                                                torch.ones(B, 1).to(torch.int64)+2, self.token_type_ids], dim=1).reshape(B*(N+2+N_))
            # token_type_table = nn.Parameter(torch.empty(4, self.d_model).normal_(std=0.02))
            one_hot_ids = torch.nn.functional.one_hot(token_type_ids, num_classes=4)
            one_hot_ids = one_hot_ids.to('mps')
            token_type_embeddings = torch.matmul(one_hot_ids.float(), self.token_type_table).reshape(B, N+2+N_, self.d_model)
            x += token_type_embeddings

        input_mask_from = torch.empty((B, N), dtype=torch.int64).bernoulli_(p=0.85)
        input_mask_to = torch.empty((B, N_), dtype=torch.int64).bernoulli_(p=0.85)
        input_mask = torch.cat([torch.ones(B, 1).to(torch.int64), input_mask_from, 
                                torch.ones(B, 1).to(torch.int64), input_mask_to], dim=1)
        inputs = create_data_from_input_mask(x.detach().cpu(), input_mask)
        attn_mask = create_attention_mask_from_input_mask((B, 2+N+N_), input_mask)
        attn_mask = attn_mask.repeat(1, 1, self.num_heads).reshape(B*self.num_heads, 2+N+N_, 2+N+N_).to(device)
        # attn_mask = -attn_mask + 1
        # attn_mask = attn_mask.to(torch.bool)
        attn_mask = torch.where(attn_mask>0, 0., -float('inf'))
        torch._assert(inputs.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {inputs.shape}")
        attn_mask = attn_mask.to('mps')
        if position_emb:
            inputs = inputs.to('mps')
            inputs += self.pos_embedding

        for i in range(self.num_layers):
            self.encoder.layers[i].attn_mask = attn_mask

        x = self.encoder(inputs)

        output_mask = -input_mask+1
        mask_cont = torch.cat([output_mask[:,1:3], output_mask[:,8:10]], dim=1).reshape(B*4)
        mask_bins = torch.cat([output_mask[:,3:6], output_mask[:,10:13]], dim=1).reshape(B*6)
        mask_bin = torch.cat([output_mask[:,6].reshape(-1,1), output_mask[:,13].reshape(-1,1)], dim=1).reshape(B*2)
        mask_cont_ = torch.nonzero(mask_cont)
        mask_bins_ = torch.nonzero(mask_bins)
        mask_bin_ = torch.nonzero(mask_bin)

        reps_cont = torch.cat([x[:,1:3,:], x[:,8:10,:]], dim=1).reshape(B*4, self.d_model)
        reps_bins = torch.cat([x[:,3:6,:], x[:,10:13,:]], dim=1).reshape(B*6, self.d_model)
        reps_bin = torch.cat([x[:,6,:], x[:,13,:]], dim=1).reshape(B*2, self.d_model)
        reps_cont_ = reps_cont[mask_cont_.squeeze()]
        reps_bins_ = reps_bins[mask_bins_.squeeze()]
        reps_bin_ = reps_bin[mask_bin_.squeeze()]

        label_conts = torch.cat([x_from[:,:6], x_to[:,:6]], dim=1).reshape(B*2*2, 3)
        label_bins = torch.cat([x_from[:,6:21], x_to[:,6:21]], dim=1).reshape(B*3*2, 5)
        label_bin = torch.cat([x_from[:,21:], x_to[:,21:]], dim=1).reshape(B*2, 4)
        label_conts_ = label_conts[mask_cont_.squeeze()]
        label_bins_ = label_bins[mask_bins_.squeeze()]
        label_bin_ = label_bin[mask_bin_.squeeze()]

        preds_conts = self.reg(reps_cont_)
        preds_bins = self.cls_bins(reps_bins_)
        preds_bin = self.cls_bin(reps_bin_)

        loss_conts = torch.nn.functional.mse_loss(preds_conts, label_conts_)
        loss_bins = torch.nn.functional.binary_cross_entropy_with_logits(preds_bins, label_bins_)
        loss_bin = torch.nn.functional.binary_cross_entropy_with_logits(preds_bin, label_bin_)
        masked_lm_loss = loss_conts + loss_bins + loss_bin

        x = x[:, 0]
        logits = self.pooler(x)
        return logits, masked_lm_loss


class PModel(nn.Module):
    def __init__(self, feature_in: int=25, hidden: int=200, **kwargs):
        super(PModel, self).__init__()
        self.feature_in = feature_in
        self.hidden = hidden
        self.pmodel = nn.Sequential(
            nn.Linear(feature_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor):
        logits = self.pmodel(x)
        return logits
    
class VModel(nn.Module):
    def __init__(self, feature_in: int=25, hidden: int=200, **kwargs):
        super().__init__()
        self.feature_in = feature_in
        self.hidden = hidden
        self.rep = nn.Sequential(
            nn.Linear(feature_in, hidden),
            nn.BatchNorm1d(hidden),
        )
        self.d0 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )
        self.d1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.rep(x)
        pred0 = self.d0(h)
        pred1 = self.d1(h)
        yf = torch.where(t>0, pred1, pred0).to(float)
        ycf = torch.where(t>0, pred0, pred1).to(float)
        return yf, ycf