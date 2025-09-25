from torch import nn
import torch

from encoders import PhoneticEncoder


class TransformBlock(nn.Module):
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError("Subclasses must implement the hidden_size property.")


class Standard(TransformBlock):
    def __init__(self, hidden_size: int, att_dim: int, **kwargs) -> None:
        super().__init__()
        self.W_e = nn.Linear(2 * hidden_size, att_dim)
        self.inner_dim = 2 * hidden_size

    @property
    def hidden_size(self) -> int:
        return self.inner_dim

    def forward(self, e, viet_texts, device):
        tilde_h = self.W_e(e)  # tilde_h shape: (batch_size, seq_len, att_dim)
        raw_enc = e
        return raw_enc, tilde_h, None


class PMAM(TransformBlock):
    def __init__(
        self,
        phonetic_enc: PhoneticEncoder,
        hidden_size: int,
        gate_hid: int,
        att_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.phonetic_enc = phonetic_enc
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + self.phonetic_enc.hidden_size, gate_hid),
            nn.Tanh(),
            nn.Linear(gate_hid, 1),
        )
        self.W_e = nn.Linear(2 * hidden_size, att_dim)
        self.W_t = nn.Linear(self.phonetic_enc.hidden_size, att_dim)
        self.U = nn.Linear(hidden_size, att_dim)
        self.v = nn.Linear(att_dim, 1)
        self.inner_dim = hidden_size

    @property
    def hidden_size(self) -> int:
        return 2 * self.inner_dim + self.phonetic_enc.hidden_size

    def forward(self, e, viet_texts, device):
        h = self.phonetic_enc(
            viet_texts, device
        )  # h shape: (batch_size, seq_len, self.phonetic_enc.hidden_size)
        eh = torch.cat(
            (e, h), dim=2
        )  # eh shape: (batch_size, seq_len, 2 * hidden_size + viet_hidden_size)
        l = self.gate_mlp(eh).squeeze(2)  # l shape: (batch_size, seq_len)
        tilde_b = torch.sigmoid(l)  # tilde_b shape: (batch_size, seq_len)

        we_e = self.W_e(e)  # we_e shape: (batch_size, seq_len, att_dim)
        x_g = (1 - tilde_b).unsqueeze(
            2
        ) * we_e  # x_g shape: (batch_size, seq_len, att_dim)
        wt_h = self.W_t(h)  # wt_h shape: (batch_size, seq_len, att_dim)

        tilde_h = x_g + wt_h  # tilde_h shape: (batch_size, seq_len, att_dim)
        raw_enc = torch.cat(
            (e, h), dim=2
        )  # raw_enc shape: (batch_size, seq_len, self.transform.hidden_size)
        return raw_enc, tilde_h, l
