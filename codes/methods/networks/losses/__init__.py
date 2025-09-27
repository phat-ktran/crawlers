from torch import nn
import torch
from torch.nn import functional as F


class DecodingLoss(nn.Module):
    def __init__(self, pad_idx: int, scale_factor: float = 1.0) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.scale = scale_factor

    def forward(self, logits, targets, l, bp, b, mask):
        return self.scale * F.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=self.pad_idx
        )


class MCGLoss(nn.Module):
    def __init__(
        self,
        pad_idx: int,
        scale_factor: float = 1.0,
        margin: float = 0.3,
        enable_contrastive=False,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.scale = scale_factor
        self.margin = margin
        self.enable_contrastive = enable_contrastive

    def forward(self, logits, targets, l, bp, b, mask):
        # ---- Drop EOS position ----
        # assume mask is (B, L) with EOS included at the last valid position
        # EOS is always the last non-pad element → zero it out
        eos_mask = mask.clone()
        eos_mask.scatter_(1, eos_mask.sum(dim=1, keepdim=True).long() - 1, 0)
    
        # ----- Binary Cross Entropy Loss (for main supervision) -----
        mtr_loss = F.binary_cross_entropy_with_logits(
            l, b.float(), reduction="none"
        )
        mtr_loss = (mtr_loss * eos_mask).sum() / eos_mask.sum()
    
        if not self.enable_contrastive:
            return self.scale * mtr_loss, {"bce": mtr_loss.item()}
    
        # ----- Contrastive Loss -----
        p = torch.sigmoid(l)  # (B, L)
        d_true = (p - b.float()).abs()
        d_approx = (p - bp.float()).abs()
        contrast_loss = (self.margin + d_approx - d_true).clamp(min=0.0)
        contrast_loss = (contrast_loss * eos_mask).sum() / eos_mask.sum()
    
        total_loss = self.scale * (mtr_loss + contrast_loss)
        return total_loss, {
            "bce": mtr_loss.item(),
            "contrast": contrast_loss.item(),
        }
