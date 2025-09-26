import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from codes.methods.utils import PAD_ID

from codes.methods.embeddings.cbow import CBOW


# ----------------------------
# Utility: masked softmax
# ----------------------------
def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9
):
    """
    logits: (..., V)
    mask: same shape as logits, 1 for allowed, 0 for disallowed
    returns: softmax probabilities with disallowed positions zeroed
    """
    neg_inf = -1e9
    logits_masked = logits.masked_fill(mask == 0, neg_inf)
    probs = F.softmax(logits_masked, dim=dim)
    # ensure zero where mask==0
    probs = probs * mask.float()
    # renormalize in case of numerical issues
    denom = probs.sum(dim=dim, keepdim=True)
    denom = denom + eps
    probs = probs / denom
    return probs


# ----------------------------
# Vectorized Model
# ----------------------------
class ConfusionPointerNetVectorized(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        enc_hidden: int = 512,
        dec_hidden: int = 512,
        attn_dim: int = 512,
        ptr_dim: int = 128,
        drop_rate: float = 0.2,
        sentinel_token_id: Optional[int] = None,
    ):
        """
        Vectorized version of ConfusionPointerNet with optimized forward pass.

        vocab_size: total number of characters
        sentinel_token_id: optional id for sentinel appended to input positions;
                          if None we will handle position distribution sized n+1 internally.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout_e = nn.Dropout(drop_rate)

        # Encoder: BiLSTM
        self.enc_hidden = enc_hidden
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=enc_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Decoder: single-layer LSTM
        self.dec_hidden = dec_hidden
        self.decoder_cell = nn.LSTMCell(input_size=embed_dim, hidden_size=dec_hidden)

        # Attention parameters (additive / Bahdanau style)
        self.W1 = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.W2 = nn.Linear(2 * enc_hidden, attn_dim, bias=False)
        self.v_att = nn.Linear(attn_dim, 1, bias=False)

        # Combine decoder hidden + context -> Cj
        self.combine = nn.Linear(dec_hidden + 2 * enc_hidden, dec_hidden)
        self.dropout_c = nn.Dropout(drop_rate)

        # vocab projection (from combined context)
        self.vocab_proj = nn.Linear(dec_hidden, vocab_size)

        # pointer network components
        self.Wg = nn.Linear(dec_hidden, dec_hidden)
        self.pointer_proj = nn.Linear(dec_hidden + 1, ptr_dim)
        self.dropout_p = nn.Dropout(drop_rate)
        self.pointer_score = nn.Linear(ptr_dim, 1)

        self.dropout = nn.Dropout(drop_rate)
        self.pad_idx = PAD_ID
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embed.weight)

        for l in [
            self.W1,
            self.W2,
            self.v_att,
            self.combine,
            self.vocab_proj,
            self.Wg,
            self.pointer_proj,
            self.pointer_score,
        ]:
            if hasattr(l, "weight"):
                nn.init.xavier_uniform_(l.weight)
            if hasattr(l, "bias") and l.bias is not None:
                nn.init.constant_(l.bias, 0.0)

        for m in [self.decoder_cell, self.encoder]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)

    def load_pretrained_embeddings(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained embeddings file not found at {path}")
        pretrained_weights = torch.load(path, map_location=torch.device("cpu"))
        cbow = CBOW(self.vocab_size - 4, self.emb_dim)
        cbow.load_state_dict(pretrained_weights)
        self.embed.weight.data[-(self.vocab_size - 4) :, :].copy_(
            cbow.in_embed.weight.data[:-1, :]
        )

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor):
        """
        src_ids: (B, n)
        src_mask: (B, n) 1 for real tokens, 0 for padding
        Returns:
            enc_hs: (B, n, 2*enc_hidden)
            enc_final: (h, c) if needed
        """
        emb = self.embed(src_ids)  # (B, n, E)
        emb = self.dropout_e(emb)
        packed_out, (h_n, c_n) = self.encoder(emb)
        enc_hs = packed_out  # (B, n, 2*enc_hidden)
        return enc_hs, (h_n, c_n)

    def batch_attention(
        self, dec_h: torch.Tensor, enc_hs: torch.Tensor, enc_mask: torch.Tensor
    ):
        """
        Vectorized attention computation.
        dec_h: (B, max_len, dec_hidden)
        enc_hs: (B, n, 2*enc_hidden)
        enc_mask: (B, n) 1 for real token, 0 for pad
        Returns:
            context: (B, max_len, 2*enc_hidden)
            attn_weights: (B, max_len, n)
        """
        B, max_len, _ = dec_h.size()
        _, n, enc_dim = enc_hs.size()

        # Expand decoder hidden states for all positions
        dec_term = (
            self.W1(dec_h).unsqueeze(2).expand(-1, -1, n, -1)
        )  # (B, max_len, n, attn_dim)
        # Expand encoder hidden states for all decoder positions
        enc_term = (
            self.W2(enc_hs).unsqueeze(1).expand(-1, max_len, -1, -1)
        )  # (B, max_len, n, attn_dim)

        # Compute attention scores
        score = self.v_att(torch.tanh(dec_term + enc_term)).squeeze(
            -1
        )  # (B, max_len, n)

        # Apply mask
        enc_mask_expanded = enc_mask.unsqueeze(1).expand(
            -1, max_len, -1
        )  # (B, max_len, n)
        score = score.masked_fill(enc_mask_expanded == 0, -1e9)
        alpha = F.softmax(score, dim=-1)  # (B, max_len, n)

        # Compute context
        enc_hs_expanded = enc_hs.unsqueeze(1).expand(
            -1, max_len, -1, -1
        )  # (B, max_len, n, enc_dim)
        context = torch.sum(
            alpha.unsqueeze(-1) * enc_hs_expanded, dim=2
        )  # (B, max_len, enc_dim)
        context = self.dropout_c(context)

        return context, alpha

    def compute_pointer_logits(
        self, combined: torch.Tensor, src_mask: torch.Tensor, max_len: int
    ):
        """
        Vectorized pointer logits computation.
        combined: (B, max_len, dec_hidden)
        src_mask: (B, n)
        Returns:
            pointer_logits: (B, max_len, n+1)
        """
        B, _, dec_hidden = combined.size()
        n = src_mask.size(1)

        # Create position indicators for all timesteps
        # Locj matrix: (max_len, n) where Locj[j, i] = 1 if i == j, 0 otherwise
        device = combined.device
        timesteps = torch.arange(max_len, device=device).unsqueeze(1)  # (max_len, 1)
        positions = torch.arange(n, device=device).unsqueeze(0)  # (1, n)
        Locj = (timesteps == positions).float()  # (max_len, n)

        # Expand for batch dimension
        Locj_batch = Locj.unsqueeze(0).expand(B, -1, -1)  # (B, max_len, n)

        # Prepare input for pointer network
        combined_exp = combined.unsqueeze(2).expand(
            -1, -1, n, -1
        )  # (B, max_len, n, dec_hidden)
        Locj_unsq = Locj_batch.unsqueeze(-1)  # (B, max_len, n, 1)
        pointer_input = torch.cat(
            [combined_exp, Locj_unsq], dim=-1
        )  # (B, max_len, n, dec_hidden+1)

        # Compute pointer scores
        pointer_hidden = torch.tanh(
            self.pointer_proj(pointer_input)
        )  # (B, max_len, n, ptr_dim)
        pointer_hidden = self.dropout_p(pointer_hidden)
        pos_scores = self.pointer_score(pointer_hidden).squeeze(-1)  # (B, max_len, n)

        # Sentinel scores
        sentinel_logits = torch.tanh(self.Wg(combined)).sum(
            dim=-1, keepdim=True
        )  # (B, max_len, 1)

        # Combine position and sentinel scores
        pointer_logits = torch.cat(
            [pos_scores, sentinel_logits], dim=-1
        )  # (B, max_len, n+1)

        # Apply padding mask
        src_mask_exp = src_mask.unsqueeze(1).expand(-1, max_len, -1)  # (B, max_len, n)
        pad_mask = torch.cat(
            [src_mask_exp, torch.ones(B, max_len, 1, device=device)], dim=2
        )  # (B, max_len, n+1)
        pointer_logits = pointer_logits.masked_fill(pad_mask == 0, -1e9)

        return pointer_logits

    def vectorized_prediction(
        self,
        pointer_logits: torch.Tensor,
        vocab_logits: torch.Tensor,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        """
        Vectorized prediction computation.
        pointer_logits: (B, max_len, n+1)
        vocab_logits: (B, max_len, V)
        src_ids: (B, n)
        src_mask: (B, n)
        Returns:
            pred_ids: (B, max_len)
        """
        B, max_len, _ = pointer_logits.size()
        n = src_ids.size(1)
        device = src_ids.device

        # Get pointer choices (argmax over pointer distribution)
        ptr_choices = pointer_logits.argmax(dim=-1)  # (B, max_len)

        # Get vocab choices (argmax over vocab distribution)
        vocab_choices = vocab_logits.argmax(dim=-1)  # (B, max_len)

        # Determine which positions should copy vs generate
        # Copy if ptr_choice < n and within valid source length
        src_lengths = src_mask.sum(dim=1)  # (B,)
        timesteps = (
            torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        )  # (B, max_len)

        copy_mask = (ptr_choices < n) & (
            timesteps < src_lengths.unsqueeze(1)
        )  # (B, max_len)

        # For copying: gather from source using pointer choices
        # Clamp ptr_choices to valid range to avoid indexing errors
        safe_ptr_choices = torch.clamp(ptr_choices, 0, n - 1)  # (B, max_len)
        copied_ids = torch.gather(
            src_ids.unsqueeze(1).expand(-1, max_len, -1),
            2,
            safe_ptr_choices.unsqueeze(-1),
        ).squeeze(-1)  # (B, max_len)

        # Combine copy and generate decisions
        pred_ids = torch.where(copy_mask, copied_ids, vocab_choices)  # (B, max_len)

        return pred_ids

    def compute_vectorized_loss(
        self,
        pointer_logits: torch.Tensor,
        vocab_logits: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_ids: torch.Tensor,
    ):
        """
        Vectorized loss computation.
        pointer_logits: (B, max_len, n+1)
        vocab_logits: (B, max_len, V)
        tgt_ids: (B, tgt_len) - includes SOS at position 0
        src_ids: (B, n)
        Returns:
            total_loss: scalar tensor
        """
        B, max_len, n_plus_1 = pointer_logits.size()
        n = n_plus_1 - 1
        device = pointer_logits.device

        # Extract targets (skip SOS at position 0)
        if tgt_ids.size(1) > 1:
            targets = tgt_ids[:, 1 : max_len + 1]  # (B, max_len)
        else:
            # Handle case where tgt_ids only has SOS
            targets = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)

        # Pad targets if necessary
        if targets.size(1) < max_len:
            pad_size = max_len - targets.size(1)
            targets = F.pad(targets, (0, pad_size), value=PAD_ID)
        elif targets.size(1) > max_len:
            targets = targets[:, :max_len]

        # Create pointer labels
        labels_pos = torch.full(
            (B, max_len), n, dtype=torch.long, device=device
        )  # default: sentinel

        # Vectorized label computation
        src_ids_expanded = src_ids.unsqueeze(1).expand(
            -1, max_len, -1
        )  # (B, max_len, n)
        targets_expanded = targets.unsqueeze(2).expand(-1, -1, n)  # (B, max_len, n)

        # Find matches between targets and source
        matches = src_ids_expanded == targets_expanded  # (B, max_len, n)

        # Get first match position for each target
        match_positions = matches.float().argmax(dim=2)  # (B, max_len)
        has_match = matches.any(dim=2)  # (B, max_len)

        # Update labels where matches exist
        labels_pos = torch.where(has_match, match_positions, labels_pos)

        # Valid positions mask (not PAD and not special tokens)
        valid_mask = (targets != PAD_ID) & (targets != 0)  # (B, max_len)

        # Compute losses only for valid positions
        total_loss = torch.tensor(0.0, device=device)

        if valid_mask.any():
            # Flatten for loss computation
            flat_valid_mask = valid_mask.reshape(-1)  # (B*max_len,)
            flat_pointer_logits = pointer_logits.reshape(-1, n_plus_1)  # (B*max_len, n+1)
            flat_labels_pos = labels_pos.reshape(-1)  # (B*max_len,)
            flat_vocab_logits = vocab_logits.reshape(-1, self.vocab_size)  # (B*max_len, V)
            flat_targets = targets.reshape(-1)  # (B*max_len,)

            valid_indices = flat_valid_mask.nonzero(as_tuple=False).squeeze(1)

            if valid_indices.numel() > 0:
                # Pointer loss for all valid positions
                pointer_loss = F.cross_entropy(
                    flat_pointer_logits[valid_indices],
                    flat_labels_pos[valid_indices],
                    reduction="mean",
                )
                total_loss = total_loss + pointer_loss

                # Vocab loss for positions that need generation (sentinel positions)
                gen_mask = (flat_labels_pos == n) & flat_valid_mask
                gen_indices = gen_mask.nonzero(as_tuple=False).squeeze(1)

                if gen_indices.numel() > 0:
                    vocab_loss = F.cross_entropy(
                        flat_vocab_logits[gen_indices],
                        flat_targets[gen_indices],
                        reduction="mean",
                    )
                    total_loss = total_loss + vocab_loss

        return total_loss

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        confusionset_mask: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Vectorized forward pass.

        src_ids: (B, n) input character ids
        src_mask: (B, n) 1/0 mask
        confusionset_mask: (B, n, V) binary mask for confusion sets
        tgt_ids: (B, max_len) target character ids starting with SOS
        teacher_forcing: whether to use teacher forcing
        """
        device = src_ids.device
        B, n = src_ids.size()

        # Encode input
        enc_hs, _ = self.encode(src_ids, src_mask)  # (B, n, 2*enc_hidden)

        # Initialize decoder states
        max_decoder_len = n
        dec_h = torch.zeros(B, self.dec_hidden, device=device)
        dec_c = torch.zeros(B, self.dec_hidden, device=device)

        # Storage for all decoder states and outputs
        all_dec_h = torch.zeros(B, max_decoder_len, self.dec_hidden, device=device)
        all_vocab_logits = torch.zeros(
            B, max_decoder_len, self.vocab_size, device=device
        )

        # Get decoder inputs
        if tgt_ids is not None and teacher_forcing:
            if tgt_ids.size(1) >= max_decoder_len + 1:
                decoder_inputs = tgt_ids[:, :max_decoder_len]  # (B, max_decoder_len)
            else:
                # Pad if necessary
                decoder_inputs = F.pad(
                    tgt_ids[:, :max_decoder_len],
                    (0, max(0, max_decoder_len - tgt_ids.size(1))),
                    value=PAD_ID,
                )
        else:
            # Start with SOS token
            decoder_inputs = torch.full(
                (B, max_decoder_len), 2, dtype=torch.long, device=device
            )

        # Run decoder for all timesteps (still sequential due to recurrent nature)
        for j in range(max_decoder_len):
            # Get input for current timestep
            if j == 0:
                current_input = decoder_inputs[:, 0]  # SOS token
            else:
                if teacher_forcing and tgt_ids is not None:
                    current_input = decoder_inputs[:, j]
                else:
                    # Use previous prediction (would need to be computed)
                    current_input = decoder_inputs[:, j]

            # Decoder step
            emb_prev = self.embed(current_input)
            emb_prev = self.dropout_e(emb_prev)
            dec_h, dec_c = self.decoder_cell(emb_prev, (dec_h, dec_c))

            # Store decoder hidden state
            all_dec_h[:, j, :] = dec_h

        # Vectorized attention computation
        context, attention_weights = self.batch_attention(all_dec_h, enc_hs, src_mask)

        # Combine decoder hidden and context
        combined = torch.tanh(
            self.combine(torch.cat([all_dec_h, context], dim=-1))
        )  # (B, max_decoder_len, dec_hidden)

        # Compute vocab logits for all timesteps
        vocab_logits = self.vocab_proj(combined)  # (B, max_decoder_len, V)

        # Apply confusion set masks
        for j in range(min(max_decoder_len, n)):
            conf_mask_j = confusionset_mask[:, j, :]  # (B, V)

            # Handle empty confusion sets
            mask_sums = conf_mask_j.sum(dim=-1, keepdim=True)  # (B, 1)
            empty_mask = mask_sums == 0
            conf_mask_j = torch.where(
                empty_mask, torch.ones_like(conf_mask_j), conf_mask_j
            )

            # Apply mask to vocab logits
            vocab_logits[:, j, :] = vocab_logits[:, j, :].masked_fill(
                conf_mask_j == 0, -1e9
            )

        # Convert to log probabilities
        all_vocab_logits = F.log_softmax(vocab_logits, dim=-1)

        # Compute pointer logits (vectorized)
        all_pointer_logits = self.compute_pointer_logits(
            combined, src_mask, max_decoder_len
        )

        # Make predictions (vectorized)
        pred_ids = self.vectorized_prediction(
            all_pointer_logits, all_vocab_logits, src_ids, src_mask
        )

        # Prepare outputs
        outputs = {
            "pred_ids": pred_ids,
            "pointer_logits": all_pointer_logits,
            "vocab_logits": all_vocab_logits,
        }

        # Compute loss if targets provided
        if tgt_ids is not None:
            loss = self.compute_vectorized_loss(
                all_pointer_logits, all_vocab_logits, tgt_ids, src_ids
            )
            outputs["loss"] = loss

        return outputs


# ----------------------------
# Utility functions
# ----------------------------
def build_confusion_mask(
    batch_src_ids: torch.Tensor, vocab_size: int, conf_map: Dict[int, List[int]]
):
    """Build confusion set mask for a batch (vectorized where possible)."""
    B, n = batch_src_ids.size()
    device = batch_src_ids.device
    mask = torch.zeros((B, n, vocab_size), dtype=torch.uint8, device=device)

    # Vectorized approach where possible
    for b in range(B):
        for i in range(n):
            cid = batch_src_ids[b, i].item()
            allowed = conf_map.get(cid, [cid])
            for a in allowed:
                if a < vocab_size:  # Safety check
                    mask[b, i, a] = 1

    return mask


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Same example as original
    vocab = ["<pad>", "这", "使", "我", "永", "生", "难", "望", "忘", "汪", "圣", "晚"]
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    pad_idx = 0

    conf_map: Dict[int, List[int]] = {
        stoi["望"]: [stoi["忘"], stoi["汪"], stoi["圣"], stoi["晚"]],
        stoi["难"]: [stoi["难"]],
    }

    # Test data
    src_sentence = ["这", "使", "我", "永", "生", "难", "望"]
    src_ids = torch.tensor([[stoi[c] for c in src_sentence]], dtype=torch.long)
    src_mask = (src_ids != pad_idx).long()

    tgt_sentence = ["这", "使", "我", "永", "生", "难", "忘"]
    tgt_ids = torch.tensor(
        [[2] + [stoi[c] for c in tgt_sentence]], dtype=torch.long
    )  # Add SOS

    conf_mask = build_confusion_mask(src_ids, V, conf_map)

    # Test vectorized model
    model = ConfusionPointerNetVectorized(
        vocab_size=23305,
        embed_dim=256,
        enc_hidden=256,
        dec_hidden=256,
        attn_dim=128,
    )

    # Test forward pass
    outputs = model(src_ids, src_mask, conf_mask, tgt_ids=tgt_ids, teacher_forcing=True)

    print("Vectorized Model Results:")
    print("Loss:", outputs["loss"].item() if "loss" in outputs else "N/A")
    print("Pred ids shape:", outputs["pred_ids"].shape)
    print("Pointer logits shape:", outputs["pointer_logits"].shape)
    print("Vocab logits shape:", outputs["vocab_logits"].shape)
