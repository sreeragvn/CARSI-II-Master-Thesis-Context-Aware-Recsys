import torch as t
from torch import nn
from config.configurator import configs

def weights_init(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight.data, -0.1, 0.1)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

class static_context_encoder(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim, hidden_dim1, hidden_dim2, output_dim):
        super(static_context_encoder, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(num_embeddings=max_val + 1, 
                                                            embedding_dim=embedding_dim) 
                                                            for max_val in vocab_sizes])
        total_embedding_dim = embedding_dim * len(vocab_sizes)
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # Batch normalization after first fully connected layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)  # Batch normalization after second fully connected layer
        self.output_layer = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)]
        embedded = t.cat(embedded, dim=1)
        fc1_out = self.fc1(embedded)
        fc1_out = self.bn1(fc1_out)  # Batch normalization after first fully connected layer
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.bn2(fc2_out)  # Batch normalization after second fully connected layer
        fc2_out = self.relu(fc2_out)
        fc2_out = self.dropout(fc2_out)
        output = self.output_layer(fc2_out)
        return output


# import torch as t
# from torch import nn
# from config.configurator import configs

# def weights_init(m):
#     if isinstance(m, nn.Embedding):
#         nn.init.uniform_(m.weight.data, -0.1, 0.1)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight.data)
#         nn.init.constant_(m.bias.data, 0.0)

# class static_context_encoder(nn.Module):
#     def __init__(self, vocab_sizes, embedding_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(static_context_encoder, self).__init__()
#         self.embedding_layers = nn.ModuleList([nn.Embedding(num_embeddings=max_val + 1, 
#                                                             embedding_dim=embedding_dim) 
#                                                             for max_val in vocab_sizes])
#         total_embedding_dim = embedding_dim * len(vocab_sizes)
#         self.fc1 = nn.Linear(total_embedding_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second fully connected layer
#         self.output_layer = nn.Linear(hidden_dim2, output_dim)
#         self.dropout = nn.Dropout(p=0.2)
#         self.relu = nn.ReLU()
#         self.apply(weights_init)

#     # def forward(self, x):
#     #     embedded = [
#     #         emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)
#     #     ]
#     #     embedded = t.cat(embedded, dim=1)  # Concatenate embeddings along dimension 1
#     #     fc1_out = t.relu(self.fc1(embedded))
#     #     fc2_out = t.relu(self.fc2(fc1_out))  # Pass through the second FC layer
#     #     output = self.output_layer(fc2_out)
#     #     return output
    
#     def forward(self, x):
#         embedded = [
#             emb(x[:, i]) for i, emb in enumerate(self.embedding_layers)
#         ]
#         embedded = t.cat(embedded, dim=1)
#         fc1_out = self.relu(self.fc1(embedded))
#         fc1_out = self.dropout(fc1_out) 
#         fc2_out = self.relu(self.fc2(fc1_out))
#         fc2_out = self.dropout(fc2_out) 
#         output = self.relu(self.output_layer(fc2_out))
#         output = self.dropout(output) 
#         return output
