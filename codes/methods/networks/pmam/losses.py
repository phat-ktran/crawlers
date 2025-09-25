from torch import nn
from torch.nn import functional as F


class DecodingLoss(nn.Module):
    def __init__(self, pad_idx: int, scale_factor: float = 1.0) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.scale = scale_factor

    def forward(self, logits, targets, l, b, mask):
        return self.scale * F.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=self.pad_idx
        )


class MCGLoss(nn.Module):
    def __init__(self, pad_idx: int, scale_factor: float = 1.0) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.scale = scale_factor

    def forward(self, logits, targets, l, b, mask):
        mtr_loss = F.binary_cross_entropy_with_logits(l, b, reduction="none")
        mtr_loss = (mtr_loss * mask).sum() / mask.sum()
        return self.scale * mtr_loss
