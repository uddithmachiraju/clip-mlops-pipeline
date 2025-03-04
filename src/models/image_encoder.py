import torch 
from torch import nn 
from pos_embeds import PositionalEncoding
from transformer_encoder import TransformerEncoder

class ImageEncoder(nn.Module):
    """
    Splitting the image into patches and create a sequence of 
    Linear embeddings of these patches by conv2d method. 
    """
    def __init__(self, hidden_dim, image_size, patch_size, n_channels, n_layers, n_heads, embedd_dim):
        super(ImageEncoder, self).__init__() 
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, \
            "Image Size Dimentions must match patch dimnetions"
        assert hidden_dim % n_heads == 0, "Hidden Dimentions must divisible by number of heads" 
        self.patches = ((image_size[0] * image_size[1]) // (patch_size[0] * patch_size[1]))
        self.max_seq_length = self.patches + 1
        self.linear_projection = nn.Conv2d(
            n_channels = n_channels, out_channels = hidden_dim, 
            kernel_size = patch_size, stride = patch_size 
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) 
        self.pos_encods = PositionalEncoding(hidden_dim, self.max_seq_length)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(hidden_dim , n_heads)
                for _ in range(n_layers) 
            ]
        )
        self.projection_layer = nn.Parameter(torch.randn(hidden_dim, embedd_dim)) 

    def forward(self, input):
        # (batch_size, color_channels, height, width) -> (batch_size, hidden_dim, patch_size, patch_size)
        input = self.linear_projection(input) 
        # (batch_size, hidden_dim, patch_size, patch_size) -> (batch_size, hidden_dim, patches) -> (batch_size, pathces, hidden_dim)
        input = input.flatten(2).transpose(1, 2) 

        # Positional Encoding
        input = torch.cat((self.cls_token.expand(input.size()[0], -1, -1), input), dim = 1)
        input = self.pos_encods(input)

        # Transformer Encoder
        for encoder_layer in self.encoder:
            input = encoder_layer(input) 

        # Getting class tokens
        input = input[:, 0, :] 

        # Join multimodel embeddings
        if self.projection_layer is not None:
            input = input @ self.projection_layer 

        input = input / torch.norm(input, dim = -1, keepdim = True)

        return input 