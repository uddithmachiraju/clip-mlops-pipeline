import torch
from torch import nn 
from src.models.pos_embeds import PositionalEncoding
from src.models.transformer_encoder import TransformerEncoder

def tokenizer(text, encode = True, mask = None, max_seq_length = 32):
    if encode:
        # Add SOT and EOT tokens
        output = chr(2) + text + chr(3) 
        # Add padding
        output = output + "".join([chr(0) for _ in range(max_seq_length - len(output))]) 
        # Encode text 
        output = torch.IntTensor(list(output.encode("utf-8")))
        # Add mask 
        mask = torch.ones(len(output.nonzero()))
        mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))) 
    else:
        # Decode the text 
        output = [chr(x) for x in text[1:len(mask.nonzero() - 1)]] 
        output = "".join(output)
        mask = None 

    return output, mask 

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_seq_length, num_heads, num_layers, embed_dim):
        super(TextEncoder, self).__init__() 
        self.max_seq_length = max_seq_length 
        self.encoder_embeds = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embeds = PositionalEncoding(hidden_dim, max_seq_length) 
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(hidden_dim, num_heads)
                for _ in range(num_layers) 
            ]
        )
        # Allign text embeddings with image embeddings
        self.projection = nn.Parameter(torch.randn(hidden_dim, embed_dim))

    def forward(self, text, mask = None):
        x = self.encoder_embeds(text)
        x = self.positional_embeds(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask) 
        x = x[
            torch.arange(text.shape[0], dtype=torch.int64),  # Ensure LongTensor
            torch.sub(torch.sum(mask[:, 0], dim=1), 1).to(torch.int64)  # Convert to int64
        ]
 

        if self.projection is not None:
            x = x @ self.projection 

        x = x / torch.norm(x, dim = -1, keepdim = True) 
        return x 