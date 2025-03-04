from torch import nn 
from attn_head import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio):
        super(TransformerEncoder, self).__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 

        # Layer Normalization 1
        self.norm_1 = nn.LayerNorm(hidden_dim)

        # Multi Head Attention
        self.attention_layer = MultiHeadAttention(hidden_dim, num_heads) 

        # Layer Normalization 2
        self.norm_2 = nn.LayerNorm(hidden_dim) 

        # Feed-Forward Layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio), # Expand Features
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim)
        )

    def forward(self, input, mask = None):
        # Apply Attention with residual connection
        input = input + self.attention_layer(self.norm_1(input), mask = mask)  

        # Apply Feed forward MLP with residual
        input = input + self.mlp(self.norm_2(input))

        return input 