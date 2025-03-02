import torch 
from torch import nn 

class AttentionHead(nn.Module):
    """
    Transformer use attention which is a communication mechanism
    that allows the model to focus on important parts of the image.

    It consists of:
        1. Query - What the token is looking for?
        2. Key - What the token contains?
        3. Value - What is communicated between tokens? 

    Attention mask is required to decoders to avoid seeing into the
    next token. Since CLIP is a encoder only model we need attention
    due to the padding that is applied to the input text during tokenization.
    """
    def __init__(self, embedd_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.head_size = head_dim 
        self.query = nn.Linear(embedd_dim, head_dim) 
        self.key = nn.Linear(embedd_dim, head_dim)
        self.value = nn.Linear(embedd_dim, head_dim) 

    def forward(self, input, mask = None):
        Q = self.query(input) 
        K = self.key(input)
        V = self.value(input) 

        # Dot product of Querys and Keys 
        attention = Q @ K.transpose(-2, -1) 

        # Scalling attention 
        attention = attention / (self.head_size ** 0.5) 

        # Apply mask to decoder(Used in testing)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        
        # Apply softmax
        attention = torch.softmax(attention, dim = -1)
        attention = attention @ V 
        return attention 
    
class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention is just multiple heads of single head
    attention in parallel and combining the output.
    """
    def __init__(self, embedd_dim, n_heads):
        super(MultiHeadAttention, self).__init__() 
        assert embedd_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.n_heads = embedd_dim // n_heads 
        self.output_layer = nn.Linear(embedd_dim, embedd_dim) 
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(embedd_dim, self.n_heads) 
                for _ in range(n_heads) 
            ]
        )

    def forward(self, input, mask= None):
        output = torch.cat(
            [
                head(input, mask) 
                for head in self.attention_heads
            ],
            dim = -1 
        )
        output = self.output_layer(output) 
        return output