import torch
from torch import nn


class QueryableTokens(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(
        self, src_enc: torch.Tensor, src_mask: torch.Tensor, pad_b: torch.Tensor
    ):
        raise NotImplementedError("This method needs to be implemented.")


class LightWeightQT(QueryableTokens):
    def __init__(
        self,
        d_model: int,
        src_hidden_size: int,
        num_heads: int = 8,
        max_len: int = 40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model

        # base learnable query token (shared)
        self.q_base = nn.Parameter(torch.randn(1, 1, d_model))

        # prototypes for binary conditioning
        self.v1 = nn.Parameter(torch.randn(1, 1, d_model))  # "likely correct"
        self.v0 = nn.Parameter(torch.randn(1, 1, d_model))  # "likely erroneous"

        # linear proj for q_base
        self.q_proj = nn.Linear(d_model, d_model)

        # positional encoding
        self.pe = nn.Parameter(torch.randn(1, max_len+1, d_model))

        # self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # cross-attention: Q from queries, K=V from encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=src_hidden_size,
            vdim=src_hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        # output classifier
        self.out = nn.Linear(d_model, 1)

    def forward(
        self, src_enc: torch.Tensor, src_mask: torch.Tensor, pad_b: torch.Tensor
    ):
        """
        src_enc: (B, N, Hs)   encoder hidden states
        src_mask: (B, N)      padding mask
        pad_b: (B, N)         binary prior flags B'
        """
        B, N, _ = src_enc.shape

        # ---- Step 1. Init queries from q_base and B' ----
        q_base_proj = self.q_proj(self.q_base).expand(B, N, -1)  # (B, N, d)
        v1 = self.v1.expand(B, N, -1)
        v0 = self.v0.expand(B, N, -1)
        queries = (
            q_base_proj + pad_b.unsqueeze(-1) * v1 + (1 - pad_b).unsqueeze(-1) * v0
        )

        # ---- Step 2. Add positional encodings ----
        queries = queries + self.pe[:, :N, :]  # (B, N, d)

        # ---- Step 3. Self-attention among queries ----
        q_self, _ = self.self_attn(
            queries, queries, queries, key_padding_mask=(src_mask == 0)
        )

        # ---- Step 4. Cross-attention with encoder ----
        q_cross, _ = self.cross_attn(
            query=q_self,
            key=src_enc,
            value=src_enc,
            key_padding_mask=(src_mask == 0),
        )

        # ---- Step 5. Project to sigmoid ----
        logits = self.out(q_cross).squeeze(-1)  # (B, N)
        if self.training:
            return logits

        probs = torch.sigmoid(logits)
        return probs


class ComplexQT(QueryableTokens):
    def __init__(
        self,
        d_model: int,
        src_hidden_size: int,
        num_heads: int = 8,
        max_len: int = 40,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Directly learnable base tokens (instead of q_base)
        self.query_tokens = nn.Parameter(torch.randn(max_len+1, d_model))

        # Prototypes for B′ conditioning
        self.v1 = nn.Parameter(torch.randn(d_model))  # "likely correct"
        self.v0 = nn.Parameter(torch.randn(d_model))  # "likely erroneous"

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_len+1, d_model))

        # Self-attention block (lightweight)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # Cross-attention to encoder embeddings (E or fused features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=src_hidden_size,
            vdim=src_hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        # Projection head
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, src_enc: torch.Tensor, src_mask: torch.Tensor, pad_b: torch.Tensor
    ):
        """
        src_enc: (B, N, H) encoder embeddings
        src_mask: (B, N) mask for encoder
        pad_b: (B, max_len) binary prior flags
        """

        B, N, H = src_enc.shape

        # Expand learnable tokens to batch
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, max_len, H)

        # Add positional encoding
        q = q + self.pos_encoding.unsqueeze(0)

        # Condition queries with B′ and prototypes
        v1 = self.v1.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        v0 = self.v0.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        q = (
            q + pad_b.unsqueeze(-1) * v1 + (1 - pad_b).unsqueeze(-1) * v0
        )  # (B, max_len, H)

        # Self-attend among queries
        q, _ = self.self_attn(q, q, q)

        # Cross-attend to encoder embeddings
        q, _ = self.cross_attn(
            query=q, key=src_enc, value=src_enc, key_padding_mask=(src_mask == 0)
        )

        # Project to binary flags
        out = self.out_proj(q).squeeze(-1)  # (B, max_len)

        return out
