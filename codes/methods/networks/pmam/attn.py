from torch import nn
import torch

from codes.methods.networks.pmam.encoders import PhoBERTEncoder


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


class CrossAttention(nn.Module):
    """
    Align Sino-Nom token stream with PhoBERT subword features
    via cross-attention.
    """
    def __init__(self, sino_hidden_size, viet_hidden_size, attn_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(sino_hidden_size, viet_hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=viet_hidden_size, num_heads=attn_heads, batch_first=True
        )
        self.out_proj = nn.Linear(viet_hidden_size, sino_hidden_size)

    def forward(self, sino_states, viet_hidden, viet_mask=None):
        """
        sino_states: (batch, seq_len, sino_hidden_size)
        viet_hidden: (batch, subword_len, viet_hidden_size)
        viet_mask:   (batch, subword_len)
        """
        Q = self.query_proj(sino_states)  # (batch, seq_len, viet_hidden_size)

        # MultiheadAttention expects key_padding_mask with True = masked
        if viet_mask is not None:
            key_padding_mask = viet_mask == 0
        else:
            key_padding_mask = None

        attn_out, attn_weights = self.attn(
            query=Q, key=viet_hidden, value=viet_hidden,
            key_padding_mask=key_padding_mask
        )
        aligned = self.out_proj(attn_out)  # (batch, seq_len, sino_hidden_size)
        return aligned, attn_weights


class PMAMWithBERT(TransformBlock):
    def __init__(
        self,
        phonetic_enc: PhoBERTEncoder,
        hidden_size: int,
        gate_hid: int,
        att_dim: int,
        att_heads: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.phonetic_enc = phonetic_enc
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + self.phonetic_enc.hidden_size, gate_hid),
            nn.Tanh(),
            nn.Linear(gate_hid, 1),
        )
        self.cross_aligner = CrossAttention(
            sino_hidden_size=2*hidden_size,
            viet_hidden_size=self.phonetic_enc.hidden_size,
            attn_heads=att_heads,
        )
        self.inner_dim = hidden_size

    @property
    def hidden_size(self) -> int:
        return 2 * self.inner_dim

    def forward(self, e, viet_texts, device):
        viet_hidden, viet_mask = self.phonetic_enc(viet_texts, device)
        aligned, attn_weights = self.cross_aligner(e, viet_hidden, viet_mask)
        
        # Concatenate and gate
        eh = torch.cat([e, aligned], dim=-1)  # (batch, seq_len, 2*sino_hidden_size)
        l = self.gate_mlp(eh).squeeze(-1)     # (batch, seq_len)
        tilde_b = torch.sigmoid(l)
        gated_enc = (1 - tilde_b).unsqueeze(-1) * e + tilde_b.unsqueeze(-1) * aligned
        
        return e, gated_enc, l