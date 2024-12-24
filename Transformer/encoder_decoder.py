from self_attention import * 


# we have used learnable positional embeddings for this implementation
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12) # eps for numerical stability
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_lenght = input_ids.size(1)
        position_ids = torch.arange(seq_lenght, dtype=torch.long).unsqueeze(0)

        token_embedings = self.token_embeddings(input_ids)
        pos_embeddings = self.pos_embeddings(position_ids)

        embeddings = token_embedings + pos_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x 
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # we will use the pre-normalization design choice
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x 
    
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers_stack = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])  

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers_stack:
            x = layer(x)
        return x
    

class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x) [:,0,:] # select the first hidden state, conventional but not restricted.
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    


