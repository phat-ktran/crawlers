from torch import nn
import torch
import logging

logger = logging.getLogger(__name__)


class Fusion(nn.Module):
    def __init__(self, src_hidden_size: int, ref_hidden_size: int, **kwargs) -> None:
        super().__init__()
        self.fused_dim = src_hidden_size

    def forward(
        self,
        src_enc: torch.Tensor,
        viet_enc: torch.Tensor,
        src_mask: torch.Tensor,
        viet_mask: torch.Tensor,
    ):
        logger.debug(f"[Fusion] pass-through: src_enc {src_enc.shape}")
        return src_enc


class Concat(Fusion):
    def __init__(self, src_hidden_size: int, ref_hidden_size: int, **kwargs) -> None:
        super().__init__(src_hidden_size, ref_hidden_size, **kwargs)
        self.fused_dim = 2 * src_hidden_size + ref_hidden_size
        self.mha = nn.MultiheadAttention(
            embed_dim=2 * src_hidden_size,
            vdim=ref_hidden_size,
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("drop_out", 0.0),
            batch_first=True,
        )

    def forward(self, src_enc, viet_enc, src_mask, viet_mask):
        logger.debug(f"[Concat] src_enc {src_enc.shape}, viet_enc {viet_enc.shape}")
        context, _ = self.mha(
            query=src_enc,
            key=viet_enc,
            value=viet_enc,
            key_padding_mask=(viet_mask == 0),
        )
        logger.debug(f"[Concat] context {context.shape}")
        fused = torch.cat([src_enc, context], dim=-1)
        logger.debug(f"[Concat] fused {fused.shape}")
        return fused


class Add(Fusion):
    def __init__(self, src_hidden_size: int, ref_hidden_size: int, **kwargs) -> None:
        super().__init__(src_hidden_size, ref_hidden_size, **kwargs)
        self.fused_dim = 2 * src_hidden_size
        assert 2 * src_hidden_size == ref_hidden_size
        self.mha = nn.MultiheadAttention(
            embed_dim=2 * src_hidden_size,
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("drop_out", 0.0),
            batch_first=True,
        )

    def forward(self, src_enc, viet_enc, src_mask, viet_mask):
        logger.debug(f"[Add] src_enc {src_enc.shape}, viet_enc {viet_enc.shape}")
        context, _ = self.mha(
            query=src_enc,
            key=viet_enc,
            value=viet_enc,
            key_padding_mask=(viet_mask == 0),
        )
        logger.debug(f"[Add] context {context.shape}")
        fused = src_enc + context
        logger.debug(f"[Add] fused {fused.shape}")
        return fused


class Sigmoid(Fusion):
    def __init__(self, src_hidden_size: int, ref_hidden_size: int, **kwargs) -> None:
        super().__init__(src_hidden_size, ref_hidden_size, **kwargs)
        num_heads = kwargs.get("num_heads", 8)
        drop_out = kwargs.get("drop_out", 0.0)

        self.mha = nn.MultiheadAttention(
            embed_dim=2 * src_hidden_size,
            num_heads=num_heads,
            dropout=drop_out,
            vdim=ref_hidden_size,
            batch_first=True,
        )
        self.gate = nn.Linear(2 * src_hidden_size + ref_hidden_size, 2 * src_hidden_size)
        self.fused_dim = 2 * src_hidden_size

    def forward(self, src_enc, viet_enc, src_mask, viet_mask):
        logger.debug(f"[Sigmoid] src_enc {src_enc.shape}, viet_enc {viet_enc.shape}")
        context, _ = self.mha(
            query=src_enc,
            key=viet_enc,
            value=viet_enc,
            key_padding_mask=(viet_mask == 0),
        )
        logger.debug(f"[Sigmoid] context {context.shape}")
        gate_inp = torch.cat([src_enc, context], dim=-1)
        g = torch.sigmoid(self.gate(gate_inp))
        logger.debug(
            f"[Sigmoid] gate {g.shape}, min {g.min().item():.4f}, max {g.max().item():.4f}"
        )
        fused = g * context + (1 - g) * src_enc
        logger.debug(f"[Sigmoid] fused {fused.shape}")
        return fused


class Residual(Fusion):
    def __init__(self, src_hidden_size: int, ref_hidden_size: int, **kwargs) -> None:
        super().__init__(src_hidden_size, ref_hidden_size, **kwargs)
        num_heads = kwargs.get("num_heads", 8)
        drop_out = kwargs.get("drop_out", 0.0)

        self.mha = nn.MultiheadAttention(
            embed_dim=2 * src_hidden_size,
            num_heads=num_heads,
            dropout=drop_out,
            vdim=ref_hidden_size,
            batch_first=True,
        )
        self.gate = nn.Linear(2 * src_hidden_size + ref_hidden_size, 2 * src_hidden_size)
        self.fused_dim = 2 * src_hidden_size

    def forward(self, src_enc, viet_enc, src_mask, viet_mask):
        logger.debug(
            f"[ResidualSigmoid] src_enc {src_enc.shape}, viet_enc {viet_enc.shape}"
        )
        context, _ = self.mha(
            query=src_enc,
            key=viet_enc,
            value=viet_enc,
            key_padding_mask=(viet_mask == 0),
        )
        logger.debug(f"[ResidualSigmoid] context {context.shape}")
        gate_inp = torch.cat([src_enc, context], dim=-1)
        g = torch.sigmoid(self.gate(gate_inp))
        logger.debug(f"[ResidualSigmoid] gate {g.shape}, mean {g.mean().item():.4f}")
        fused = src_enc + g * context
        logger.debug(f"[ResidualSigmoid] fused {fused.shape}")
        return fused


class AdaptiveTemporalPooling(Fusion):
    def __init__(self, src_hidden_size: int, ref_hidden_size: int, **kwargs):
        super().__init__(src_hidden_size, ref_hidden_size, **kwargs)

        method = kwargs.get("method", "mean")
        assert method in ["mean", "max"]

        self.method = method
        self.mode = kwargs.get("mode", "Concat")  # concat | add

        if self.mode == "Concat":
            self.fused_dim = src_hidden_size + ref_hidden_size
        elif self.mode == "Add":
            assert 2 * src_hidden_size == ref_hidden_size, \
                "For add, src_hidden_size must == ref_hidden_size"
            self.fused_dim = 2 * src_hidden_size
        else:
            raise ValueError("mode must be concat or add")

    def forward(self, src_enc, viet_enc, src_mask, viet_mask):
        """
        src_enc: (B, N, Hs) - encoder hidden states
        viet_encs: (B, M, Hr) - BERT subword hidden states
        N: int, target number of pooled embeddings (same as len of src_enc)
        """
        B, M, Hr = viet_enc.shape
        _, N, _ = src_enc.shape
        logger.debug(f"[AdaptivePooling] src_enc {src_enc.shape}, viet_encs {viet_enc.shape}, target N={N}")
        

        # Transpose to (B, Hr, M) for pooling over time
        viet_encs_t = viet_enc.transpose(1, 2)

        if self.method == "mean":
            pool = nn.AdaptiveAvgPool1d(N)
        else:
            pool = nn.AdaptiveMaxPool1d(N)

        # (B, Hr, N) → transpose back → (B, N, Hr)
        viet_pooled = pool(viet_encs_t).transpose(1, 2)

        # Fuse with src_enc
        if self.mode == "Concat":
            fused = torch.cat([src_enc, viet_pooled], dim=-1)  # (B, N, Hs+Hr)
        else:
            fused = src_enc + viet_pooled  # (B, N, Hs)

        return fused