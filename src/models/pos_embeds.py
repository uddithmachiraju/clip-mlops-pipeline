import torch 
import numpy as np 
from torch import nn 

class PositionalEncoding(nn.Module):
    """
    Unlike models like LSTM's which take embeddings in sequentially,
    transformers take the embeddings in parallel. Transformers are 
    not aware of what order the sequenences are supposed t be in. This 
    will be a problem because changing the order of the sequnece will
    alter it's meaning. So, positional encoding need to be added to the 
    embeddings. Each positional encodinng is unique with it's position
    that it represents which allows model to identify which position
    each embed should go in.
    """
    def __init__(self, hidden_dim, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, hidden_dim) 

        for position in range(max_seq_length):
            for i in hidden_dim:
                if position % 2 == 0:
                    pe[position][i] = np.sin(position / (10000 ** (i / hidden_dim))) 
                else:
                    pe[position][i] = np.cos(position / (10000 ** ((i - 1) / hidden_dim))) 

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, input):
        input = input + self.pe 
        return input 