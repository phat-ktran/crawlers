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

    def forward(self, sino_states, viet_hidden, viet_mask=None, sino_mask=None):
        """
        sino_states: (batch, seq_len, sino_hidden_size)
        viet_hidden: (batch, subword_len, viet_hidden_size)
        viet_mask:   (batch, subword_len)
        sino_mask:   (batch, seq_len)  # New: Mask for Sino-Nom queries (1=valid, 0=padded)
        """
        Q = self.query_proj(sino_states)  # (batch, seq_len, viet_hidden_size)

        # Handle key padding mask
        if viet_mask is not None:
            key_padding_mask = viet_mask == 0
        else:
            key_padding_mask = None

        # Handle query padding: Create an additive mask to ignore padded queries
        if sino_mask is not None:
            batch_size, query_len, key_len = sino_states.size(0), sino_states.size(1), viet_hidden.size(1)
            attn_mask = torch.zeros(batch_size, query_len, key_len, device=sino_states.device)
            # Mask entire rows for padded queries
            attn_mask = attn_mask.masked_fill((sino_mask == 0).unsqueeze(-1), float('-inf'))
            # Expand to (batch_size * num_heads, query_len, key_len)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.attn.num_heads, -1, -1).reshape(
                batch_size * self.attn.num_heads, query_len, key_len
            )
        else:
            attn_mask = None

        attn_out, attn_weights = self.attn(
            query=Q, key=viet_hidden, value=viet_hidden,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask 
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

    def forward(self, e, viet_texts, device, mask=None):
        """
        e:           (batch, seq_len, sino_hidden_size)  # Encoder states
        viet_texts:  Input texts for PhoBERT
        device:      Torch device
        mask:        (batch, seq_len)  # New: Mask for encoder stream (1=valid, 0=padded)
        """
        viet_hidden, viet_mask = self.phonetic_enc(viet_texts, device)
        aligned, attn_weights = self.cross_aligner(e, viet_hidden, viet_mask, sino_mask=mask)
        
        # Concatenate and gate
        eh = torch.cat([e, aligned], dim=-1)  # (batch, seq_len, 2*sino_hidden_size + self.phonetic_enc.hidden_size)
        l = self.gate_mlp(eh).squeeze(-1)     # (batch, seq_len)
        
        # Apply mask to logits: Set masked positions to large negative for sigmoid -> 0
        if mask is not None:
            l = l.masked_fill(mask == 0, float('-1e9'))
        
        tilde_b = torch.sigmoid(l)
                
        gated_enc = (1 - tilde_b).unsqueeze(-1) * e + tilde_b.unsqueeze(-1) * aligned
        
        return e, gated_enc, l