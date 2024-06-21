import torch
import torch.nn as nn

######################### RNN, LSTM, GRU

class RNN(nn.Module):
    def __init__(self, input_size_cont, input_size_cat, hidden_size, num_classes, num_layers, num_embedding, model):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = model
        input_size_cat
        if model == "rnn":
            self.seq = nn.RNN(input_size_cont, hidden_size, num_layers, batch_first=True)
        elif model == "gru":
            self.seq = nn.GRU(input_size_cont, hidden_size, num_layers, batch_first=True)
        elif model == "lstm":
            self.seq = nn.LSTM(input_size_cont, hidden_size, num_layers, batch_first=True)
        
        out_emb = 64 
        self.emb = nn.Embedding(num_embedding, out_emb)

        self.fc1 = nn.Linear(hidden_size + out_emb*input_size_cat, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x_cont, x_static):

        out_emb = self.emb(x_static)
        out_emb = out_emb.view(x_static.size(0), -1)

        # Sequential
        h0 = torch.zeros(self.num_layers, x_cont.size(0), self.hidden_size)
        if self.model == "lstm":
            c0 = torch.zeros(self.num_layers, x_cont.size(0), self.hidden_size)
            out_seq, _ = self.seq(x_cont, (h0, c0))
        else:
            out_seq, _ = self.seq(x_cont, h0)
        out_seq = out_seq[:,-1,:]

        out = torch.cat((out_seq, out_emb), dim=1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

######################### CARSI

class CARSI(nn.Module):
    def __init__(
            self, 
            input_size_cont, # num_features_continuous
            input_size_cat,
            output_size,
            seq_len,
            num_embedding,
            hidden_dim=512, # d_model
            num_heads=8,
    ):
        super(CARSI, self).__init__()

        feed_forward_size = 1024

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        input_size_cat 

        # Use linear layer instead of embedding 
        self.input_embedding = nn.Linear(input_size_cont, hidden_dim)
        self.pos_enc = self.positional_encoding()

        # Multi-Head Attention
        self.multihead = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        # position-wise Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, hidden_dim)
        )

        self.fc_out1 = nn.Linear(hidden_dim, 64)

        # Embedding for static
        out_emb = 64
        self.emb = nn.Embedding(num_embedding, out_emb)

        self.fc1 = nn.Linear(64*seq_len + out_emb*input_size_cat, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)


    def positional_encoding(self):
        pe = torch.zeros(self.seq_len, self.hidden_dim) # positional encoding 
        pos = torch.arange(0, self.seq_len, dtype=torch.float32).unsqueeze(1)
        _2i = torch.arange(0, self.hidden_dim, step=2).float()
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.hidden_dim)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.hidden_dim)))
        return pe

    def forward(self, x_cont, x_static):

        # Static variables
        out_emb = self.emb(x_static)
        out_emb = out_emb.view(x_static.size(0), -1)
        
        # Embedding + Positional
        x = self.input_embedding(x_cont)
        x += self.pos_enc

        # Multi-Head Attention
        x_, _ = self.multihead(x,x,x)
        x_ = self.dropout_1(x_)

        # Add and Norm 1
        x = self.layer_norm_1(x_ + x)

        # Feed Forward
        x_ = self.feed_forward(x)
        x_ = self.dropout_2(x_)

        # Add and Norm 2
        x = self.layer_norm_2(x_ + x)

        # Output (customized flatten)
        x = self.fc_out1(x)
        # shape: N, num_features, 64
        x = torch.flatten(x, start_dim=1)
        
        # Combine static and continupus
        out = torch.cat((x, out_emb), dim=1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out

if __name__ == "__main__":

    # shape: batch, seq_len, num_features
    sample = torch.rand(4, 30, 6)

    _, seq_len, num_features = sample.size()

    num_classes = 20

    model = CARSI(
        input_size= num_features,
        output_size= num_classes,
        seq_len= seq_len,
    )

    model(sample)