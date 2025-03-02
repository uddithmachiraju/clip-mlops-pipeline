import torch 
import pytest 
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.attn_head import AttentionHead, MultiHeadAttention

@pytest.mark.parametrize("embedding_dim, head_dim, seq_length, batch_size", [
    (32, 8, 10, 2),
    (64, 16, 20, 4),
    (128, 32, 5, 1),
])

def test_attention_output_shape(embedding_dim, head_dim, seq_length, batch_size):
    """Test if the attention returns the correct output shape"""
    model = AttentionHead(embedding_dim, head_dim) 
    inputs = torch.randn(batch_size, seq_length, embedding_dim) 
    output = model(inputs) 
    assert output.shape == (batch_size, seq_length, head_dim), \
        f"Expected Output shape ({batch_size}, {seq_length}, {head_dim}) got {output.shape}" 

@pytest.mark.parametrize("embedding_dim, num_heads, seq_length, batch_size", [
    (32, 4, 10, 2),
    (64, 8, 20, 4),
    (128, 16, 5, 1),
])

def test_multihead_output_shape(embedding_dim, num_heads, seq_length, batch_size):
    """Test if Multi Head Attention returns the correct output"""
    model = MultiHeadAttention(embedding_dim, num_heads) 
    inputs = torch.randn(batch_size, seq_length, embedding_dim) 
    output = model(inputs)

    assert output.shape == (batch_size, seq_length, embedding_dim), \
    f"Excepted Shape ({batch_size}, {seq_length}, {embedding_dim}) got {output.shape}" 

def test_attention_masking():
    """Test if the attention mask correctly prevents attending to masked tokens"""
    pass 
    