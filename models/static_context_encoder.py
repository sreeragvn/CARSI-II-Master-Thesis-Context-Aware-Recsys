import torch as t
from torch import nn
from config.configurator import configs

class static_context_encoder(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim, hidden_dim1, hidden_dim2, output_dim):
        super(static_context_encoder, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(num_embeddings=max_val + 1, 
                                                            embedding_dim=embedding_dim) 
                                                            for max_val in vocab_sizes])
        total_embedding_dim = embedding_dim * len(vocab_sizes)
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second fully connected layer
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        embedded = [
            emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)
        ]
        embedded = t.cat(embedded, dim=1)  # Concatenate embeddings along dimension 1
        fc1_out = t.relu(self.fc1(embedded))
        fc2_out = t.relu(self.fc2(fc1_out))  # Pass through the second FC layer
        output = self.output_layer(fc2_out)
        return output

