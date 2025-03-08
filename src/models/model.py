import torch 
from torch import nn 
import numpy as np 
from image_encoder import ImageEncoder 
from text_encoder import TextEncoder 

class CLIP(nn.Module):
    def __init__(self, embedd_dim, vit_dim, image_size, 
                 patch_size, n_channels, vit_layers, 
                 vit_heads, vocab_size, text_dim, max_sequence, 
                 text_heads, text_layers):
        super().__init__()
        self.image_encoder = ImageEncoder(
            vit_dim, image_size, patch_size, 
            n_channels, vit_layers, vit_heads, embedd_dim
        )

        self.text_encoder = TextEncoder(
            vocab_size, text_dim, max_sequence, 
            text_heads, text_layers, embedd_dim
        )

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 /0.07))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    def forward(self, image, text, mask = None):
        image_e = self.image_encoder(image)
        text_e = self.text_encoder(text, mask = mask) 

        logits = (image_e @ text_e.transpose(-2, -1)) * torch.exp(self.temperature) 

        labels = torch.arange(logits.shape[0]).to(device=self.device) 

        loss_image = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_text = nn.functional.cross_entropy(logits, labels) 

        loss = (loss_image + loss_text) / 2 

        return loss 