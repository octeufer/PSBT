import torch
import numpy as np
import random
from einops import rearrange

def create_data_from_input_mask(data: torch.Tensor, input_mask: torch.Tensor):
    B, N, D = data.size()
    data = rearrange(data, 'b n d -> (b n) d')
    indexes = torch.arange(B*N)
    input_mask = rearrange(input_mask, 'b n -> (b n)')
    def rmap(a, b):
        if b==0:
            flag = random.random()
            if flag < 0.8:
                return 0
            elif flag < 0.9:
                return random.randint(1, B*N-1)
            else:
                return a
        else:
            return a
    vrmap = np.vectorize(rmap)
    indexes_ = vrmap(indexes, input_mask)
    data = data[indexes_]
    data = rearrange(data,'(b n) d -> b n d', b=B)
    return data

def create_attention_mask_from_input_mask(from_shape, to_mask):
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    
    to_shape = to_mask.size()
    to_seq_length = to_shape[1]

    to_mask = to_mask.reshape([batch_size, 1, to_seq_length]).to(torch.float32)
    broadcast_ones = torch.ones((batch_size, from_seq_length, 1), dtype=torch.float32)

    mask = broadcast_ones * to_mask
    # mask = -mask + 1
    # mask = mask.to(torch.bool)
    # mask = torch.where(mask>0, 0.0, -float('inf'))
    return mask


def validation_split(n, val_fraction):
    """ Construct a train/validation split """
    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid