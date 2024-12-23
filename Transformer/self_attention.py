import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value):
    dim_k = key.size(-1) # the embeddings length
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1) # along columns of the score matrix
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim) # (1, embed_dim) x (embed_dim, head_dim) == (1, head_dim) project the embeddings onto a different space
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)


    def forward(self, hidden_state):
        attn_output = scaled_dot_product_attention(self.query(hidden_state), self.key(hidden_state), self.value(hidden_state))
        return attn_output
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        ) 
        self.output_layer = nn.Linear(embed_dim, embed_dim) # the output shape is the same as the input

    def forward(self, hidden_state):
        x = torch.concat([h(hidden_state) for h in self.heads], dim = -1) # concatenate on the column direction
        x = self.output_layer(x)
        return x 

