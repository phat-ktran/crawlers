from torch import nn
import torch
import logging

logger = logging.getLogger(__name__)


class DecFusion(nn.Module):
    def __init__(
        self, src_ctx_hidden_size: int, ref_ctx_hidden_size: int, **kwargs
    ) -> None:
        super().__init__()
        self.fused_dim = src_ctx_hidden_size

    def forward(self, src_ctx: torch.Tensor, ref_ctx: torch.Tensor):
        logger.debug(f"[Fusion] pass-through: src_ctx {src_ctx.shape}")
        return src_ctx


class DecConcat(DecFusion):
    def __init__(
        self, src_ctx_hidden_size: int, ref_ctx_hidden_size: int, **kwargs
    ) -> None:
        super().__init__(src_ctx_hidden_size, ref_ctx_hidden_size, **kwargs)
        self.fused_dim = 2 * src_ctx_hidden_size + ref_ctx_hidden_size

    def forward(self, src_ctx, ref_ctx):
        logger.debug(f"[DecConcat] src_ctx {src_ctx.shape}, ref_ctx {ref_ctx.shape}")
        fused = torch.cat([src_ctx, ref_ctx], dim=-1)
        logger.debug(f"[DecConcat] fused {fused.shape}")
        return fused


class DecAdd(DecFusion):
    def __init__(
        self, src_ctx_hidden_size: int, ref_ctx_hidden_size: int, **kwargs
    ) -> None:
        super().__init__(src_ctx_hidden_size, ref_ctx_hidden_size, **kwargs)
        self.fused_dim = 2 * src_ctx_hidden_size
        assert 2 * src_ctx_hidden_size == ref_ctx_hidden_size

    def forward(self, src_ctx, ref_ctx):
        logger.debug(f"[DecAdd] src_ctx {src_ctx.shape}, ref_ctx {ref_ctx.shape}")
        fused = src_ctx + ref_ctx
        logger.debug(f"[DecAdd] fused {fused.shape}")
        return fused


class Sigmoid(DecFusion):
    def __init__(
        self, src_ctx_hidden_size: int, ref_ctx_hidden_size: int, **kwargs
    ) -> None:
        super().__init__(src_ctx_hidden_size, ref_ctx_hidden_size, **kwargs)
        num_heads = kwargs.get("num_heads", 8)
        drop_out = kwargs.get("drop_out", 0.0)

        self.mha = nn.MultiheadAttention(
            embed_dim=2 * src_ctx_hidden_size,
            num_heads=num_heads,
            dropout=drop_out,
            vdim=ref_ctx_hidden_size,
            batch_first=True,
        )
        self.gate = nn.Linear(
            2 * src_ctx_hidden_size + ref_ctx_hidden_size, 2 * src_ctx_hidden_size
        )
        self.fused_dim = 2 * src_ctx_hidden_size

    def forward(self, src_ctx, ref_ctx):
        logger.debug(f"[Sigmoid] src_ctx {src_ctx.shape}, ref_ctx {ref_ctx.shape}")
        context, _ = self.mha(
            query=src_ctx,
            key=ref_ctx,
            value=ref_ctx,
            key_padding_mask=(ref_mask == 0),
        )
        logger.debug(f"[Sigmoid] context {context.shape}")
        gate_inp = torch.cat([src_ctx, context], dim=-1)
        g = torch.sigmoid(self.gate(gate_inp))
        logger.debug(
            f"[Sigmoid] gate {g.shape}, min {g.min().item():.4f}, max {g.max().item():.4f}"
        )
        fused = g * context + (1 - g) * src_ctx
        logger.debug(f"[Sigmoid] fused {fused.shape}")
        return fused


class Residual(DecFusion):
    def __init__(
        self, src_ctx_hidden_size: int, ref_ctx_hidden_size: int, **kwargs
    ) -> None:
        super().__init__(src_ctx_hidden_size, ref_ctx_hidden_size, **kwargs)
        num_heads = kwargs.get("num_heads", 8)
        drop_out = kwargs.get("drop_out", 0.0)

        self.mha = nn.MultiheadAttention(
            embed_dim=2 * src_ctx_hidden_size,
            num_heads=num_heads,
            dropout=drop_out,
            vdim=ref_ctx_hidden_size,
            batch_first=True,
        )
        self.gate = nn.Linear(
            2 * src_ctx_hidden_size + ref_ctx_hidden_size, 2 * src_ctx_hidden_size
        )
        self.fused_dim = 2 * src_ctx_hidden_size

    def forward(self, src_ctx, ref_ctx):
        logger.debug(
            f"[ResidualSigmoid] src_ctx {src_ctx.shape}, ref_ctx {ref_ctx.shape}"
        )
        context, _ = self.mha(
            query=src_ctx,
            key=ref_ctx,
            value=ref_ctx,
            key_padding_mask=(ref_mask == 0),
        )
        logger.debug(f"[ResidualSigmoid] context {context.shape}")
        gate_inp = torch.cat([src_ctx, context], dim=-1)
        g = torch.sigmoid(self.gate(gate_inp))
        logger.debug(f"[ResidualSigmoid] gate {g.shape}, mean {g.mean().item():.4f}")
        fused = src_ctx + g * context
        logger.debug(f"[ResidualSigmoid] fused {fused.shape}")
        return fused


class AdaptiveTemporalPooling(DecFusion):
    def __init__(self, src_ctx_hidden_size: int, ref_ctx_hidden_size: int, **kwargs):
        super().__init__(src_ctx_hidden_size, ref_ctx_hidden_size, **kwargs)

        method = kwargs.get("method", "mean")
        assert method in ["mean", "max"]

        self.method = method
        self.mode = kwargs.get("mode", "Concat")  # concat | add

        if self.mode == "Concat":
            self.fused_dim = src_ctx_hidden_size + ref_ctx_hidden_size
        elif self.mode == "Add":
            assert 2 * src_ctx_hidden_size == ref_ctx_hidden_size, (
                "For add, src_ctx_hidden_size must == ref_ctx_hidden_size"
            )
            self.fused_dim = 2 * src_ctx_hidden_size
        else:
            raise ValueError("mode must be concat or add")

    def forward(self, src_ctx, ref_ctx):
        """
        src_ctx: (B, N, Hs) - encoder hidden states
        ref_ctxs: (B, M, Hr) - BERT subword hidden states
        N: int, target number of pooled embeddings (same as len of src_ctx)
        """
        B, M, Hr = ref_ctx.shape
        _, N, _ = src_ctx.shape
        logger.debug(
            f"[AdaptivePooling] src_ctx {src_ctx.shape}, ref_ctxs {ref_ctx.shape}, target N={N}"
        )

        # Transpose to (B, Hr, M) for pooling over time
        ref_ctxs_t = ref_ctx.transpose(1, 2)

        if self.method == "mean":
            pool = nn.AdaptiveAvgPool1d(N)
        else:
            pool = nn.AdaptiveMaxPool1d(N)

        # (B, Hr, N) → transpose back → (B, N, Hr)
        viet_pooled = pool(ref_ctxs_t).transpose(1, 2)

        # Fuse with src_ctx
        if self.mode == "Concat":
            fused = torch.cat([src_ctx, viet_pooled], dim=-1)  # (B, N, Hs+Hr)
        else:
            fused = src_ctx + viet_pooled  # (B, N, Hs)

        return fused
