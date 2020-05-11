import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):

    def __init__(self,
                vocab_size,
                embedding_dim,
                hidden_dim,
                num_layers=1,
                bidirectional=True):
        super(LSTMEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

        pass
    
    def forward(self, x):
        emb = self.embedding_layer(x)

        output, _ = self.lstm(emb)

        return output

class LSTMDecoder(nn.Module):

    def __init__(self,
                vocab_size,
                embedding_dim,
                hidden_dim,
                num_layers=1,
                bidirectional=True):
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, vocab_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, h_0, c_0):
        embd = self.embedding_layer(x)

        _, (h_n, c_n) = self.lstm(embd, (h_0, c_0))

        output = self.output(h_n[0])

        return output, h_n, c_n 
        

        
        