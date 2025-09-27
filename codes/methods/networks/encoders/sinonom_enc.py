from torch import nn

class SinoNomEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, 1, bidirectional=True, batch_first=True
        )

    def init_weights(self):
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)  # input -> hidden
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)  # hidden -> hidden
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def forward(self, embed_x):
        return self.encoder(embed_x)
    