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
# Model
# ----------------------------
class ConfusionPointerNet(nn.Module):
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
        # map decoder hidden -> attn_dim, encoder hidden (2*enc_hidden) -> attn_dim
        self.W1 = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.W2 = nn.Linear(2 * enc_hidden, attn_dim, bias=False)
        self.v_att = nn.Linear(attn_dim, 1, bias=False)  # scalar score

        # Combine decoder hidden + context -> Cj
        self.combine = nn.Linear(dec_hidden + 2 * enc_hidden, dec_hidden)
        self.dropout_c = nn.Dropout(drop_rate)

        # vocab projection (from combined context)
        self.vocab_proj = nn.Linear(dec_hidden, vocab_size)

        # pointer network components
        # Wi and Wg in paper: make a small MLP that takes [Wg*Cj ; Locj]
        # We implement Wg as a linear on Cj
        self.Wg = nn.Linear(dec_hidden, dec_hidden)

        # Loc embedding isn't necessary as Loc is one-hot; instead we will concat one-hot Loc and Cj and project
        self.pointer_proj = nn.Linear(dec_hidden + 1, ptr_dim)
        self.dropout_p = nn.Dropout(drop_rate)
        self.pointer_score = nn.Linear(ptr_dim, 1)

        # optional dropout
        self.dropout = nn.Dropout(drop_rate)

        # sentinel handling is internal; no external token needed
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
                    nn.init.xavier_uniform_(param)  # input -> hidden
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)  # hidden -> hidden
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # optional: set forget-gate bias to 1 (common trick for LSTMs)
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
        packed_out, (h_n, c_n) = self.encoder(emb)  # packed_out: (B, n, 2*enc_hidden)
        # encoder is batch_first=True so packed_out is actually the outputs
        enc_hs = packed_out  # (B, n, 2*enc_hidden)
        return enc_hs, (h_n, c_n)

    def attention(
        self, dec_h: torch.Tensor, enc_hs: torch.Tensor, enc_mask: torch.Tensor
    ):
        """
        dec_h: (B, dec_hidden)
        enc_hs: (B, n, 2*enc_hidden)
        enc_mask: (B, n) 1 for real token, 0 for pad
        Returns:
            context: (B, 2*enc_hidden)
            attn_weights: (B, n)
        """
        # shape preparation
        B, n, _ = enc_hs.size()
        # compute scores
        dec_term = self.W1(dec_h).unsqueeze(1).expand(-1, n, -1)  # (B, n, attn_dim)
        enc_term = self.W2(enc_hs)  # (B, n, attn_dim)
        score = self.v_att(torch.tanh(dec_term + enc_term)).squeeze(-1)  # (B, n)
        # mask pads
        score = score.masked_fill(enc_mask == 0, -1e9)
        alpha = F.softmax(score, dim=-1)  # (B, n)
        # compute context
        context = torch.bmm(alpha.unsqueeze(1), enc_hs).squeeze(1)  # (B, 2*enc_hidden)
        context = self.dropout_c(context)
        return context, alpha

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        confusionset_mask: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        src_ids: (B, n) input character ids
        src_mask: (B, n) 1/0 mask
        confusionset_mask: (B, n, V) binary mask: for each position i in input, which vocab ids belong to its confusionset.
                           NOTE: confusionset_mask[b, i] is a length-V binary mask for position i of sample b.
        tgt_ids: (B, max_len) target character ids starting with SOS (for teacher forcing)
        teacher_forcing: if True (training) feed gold prev tokens to decoder; if False (inference) use predicted token
        Returns dict containing:
            'loss': scalar tensor if tgt_ids provided
            'pred_ids': (B, n) predicted ids (greedy)
            'pointer_logits': (B, n, n+1) pointer logits per timestep
            'vocab_logits': (B, n, V) vocab logits per timestep
        """
        device = src_ids.device
        B, n = src_ids.size()
        enc_hs, _ = self.encode(src_ids, src_mask)  # (B, n, 2*enc_hidden)

        # initialize decoder hidden state: simplest is zeros; more advanced: project encoder final states
        dec_h = torch.zeros(B, self.dec_hidden, device=device)
        dec_c = torch.zeros(B, self.dec_hidden, device=device)

        # Get max decoder length - use src length for alignment
        max_decoder_len = n

        # Start with SOS token (or first target token if available)
        if tgt_ids is not None and teacher_forcing:
            decoder_input = tgt_ids[:, 0]  # Start with SOS from target
        else:
            decoder_input = torch.full(
                (B,), 2, dtype=torch.long, device=device
            )  # SOS_ID = 2

        all_vocab_logits = torch.zeros(
            B, max_decoder_len, self.vocab_size, device=device
        )
        all_pointer_logits = torch.zeros(B, max_decoder_len, n + 1, device=device)
        all_predict_words = []

        total_loss = torch.tensor(0.0, device=device)

        for j in range(max_decoder_len):
            # prepare input embedding
            emb_prev = self.embed(decoder_input)  # (B, E)
            emb_prev = self.dropout_e(emb_prev)

            dec_h, dec_c = self.decoder_cell(
                emb_prev, (dec_h, dec_c)
            )  # each (B, dec_hidden)

            # attention
            context, alpha = self.attention(
                dec_h, enc_hs, src_mask
            )  # context: (B, 2*enc_hidden)

            # combine to produce Cj (paper: Cj = tanh(W (ht_j ; htj'))
            combined = torch.tanh(
                self.combine(torch.cat([dec_h, context], dim=-1))
            )  # (B, dec_hidden)

            # --- Vocab logits ---
            vocab_logits = self.vocab_proj(combined)  # (B, V)

            # Apply confusion set mask if position j is valid
            if j < n:
                # Create masked vocab distribution
                conf_mask_j = confusionset_mask[:, j, :]  # (B, V)
                # Create a default mask for samples where confusion mask is all zeros
                mask_sums = conf_mask_j.sum(dim=-1)  # (B,)
                for b in range(B):
                    if mask_sums[b] == 0:
                        conf_mask_j[b, :] = 1  # Allow all vocab if no confusion set

                # Apply mask by setting disallowed positions to -inf
                masked_vocab_logits = vocab_logits.masked_fill(conf_mask_j == 0, -1e9)
                vocab_probs = F.log_softmax(masked_vocab_logits, dim=-1)
            else:
                vocab_probs = F.log_softmax(vocab_logits, dim=-1)

            all_vocab_logits[:, j, :] = vocab_probs

            # --- Pointer logits ---
            # create Locj: vector with 1 at position j
            Locj = torch.zeros(B, n, device=device)
            if j < n:
                Locj[:, j] = 1.0

            # Compute pointer distribution
            Locj_unsq = Locj.unsqueeze(-1)  # (B, n, 1)
            Cj_exp = combined.unsqueeze(1).expand(-1, n, -1)
            pointer_input = torch.cat(
                [Cj_exp, Locj_unsq], dim=-1
            )  # (B, n, dec_hidden+1)

            pointer_hidden = torch.tanh(
                self.pointer_proj(pointer_input)
            )  # (B, n, ptr_dim)
            pointer_hidden = self.dropout_p(pointer_hidden)
            pos_scores = self.pointer_score(pointer_hidden).squeeze(-1)  # (B, n)

            # sentinel score
            sentinel_logit = torch.tanh(self.Wg(combined)).sum(
                dim=-1, keepdim=True
            )  # (B, 1)
            pointer_logits = torch.cat([pos_scores, sentinel_logit], dim=-1)  # (B, n+1)

            # mask padded positions
            pad_mask = torch.cat(
                [src_mask, torch.ones(B, 1, device=device)], dim=1
            )  # (B, n+1)
            pointer_logits = pointer_logits.masked_fill(pad_mask == 0, -1e9)
            F.softmax(pointer_logits, dim=-1)

            all_pointer_logits[:, j, :] = pointer_logits

            # --- Make prediction for next step ---
            # Pointer decision: copy from source or generate
            ptr_choice = pointer_logits.argmax(dim=-1)  # (B,)
            vocab_choice = vocab_probs.argmax(dim=-1)  # (B,)

            # Decide based on pointer distribution
            next_ids = torch.zeros_like(ptr_choice)
            for b in range(B):
                if ptr_choice[b] < n and j < src_mask[b].sum():  # copy from source
                    pos = ptr_choice[b].item()
                    next_ids[b] = src_ids[b, pos]
                else:  # generate from vocab
                    next_ids[b] = vocab_choice[b]

            # Convert to word list (for compatibility)
            batch_words = [str(next_ids[b].item()) for b in range(B)]
            all_predict_words.append(batch_words)

            # --- Loss computation ---
            if tgt_ids is not None:
                # Target for this timestep (j+1 because tgt_ids starts with SOS)
                if j + 1 < tgt_ids.size(1):
                    target_j = tgt_ids[:, j + 1]  # (B,)

                    # Compute pointer labels
                    labels_pos = torch.full(
                        (B,), n, dtype=torch.long, device=device
                    )  # default: sentinel

                    for b in range(B):
                        if target_j[b] != PAD_ID and target_j[b] != 0:  # valid target
                            # Find if target exists in source
                            matches = (src_ids[b] == target_j[b]).nonzero(
                                as_tuple=False
                            )
                            if matches.numel() > 0:
                                labels_pos[b] = matches[0, 0].item()  # use first match
                            # else: keep sentinel (n)

                    # Pointer loss
                    valid_mask = target_j != PAD_ID
                    if valid_mask.any():
                        pointer_loss_j = F.cross_entropy(
                            pointer_logits[valid_mask],
                            labels_pos[valid_mask],
                            reduction="mean",
                        )

                        # Vocab loss (for positions that need generation)
                        gen_mask = (labels_pos == n) & valid_mask
                        if gen_mask.any():
                            vocab_loss_j = F.cross_entropy(
                                vocab_probs[gen_mask],
                                target_j[gen_mask],
                                reduction="mean",
                            )
                        else:
                            vocab_loss_j = torch.tensor(0.0, device=device)

                        total_loss = total_loss + pointer_loss_j + vocab_loss_j

            # --- Teacher forcing for next input ---
            if teacher_forcing and tgt_ids is not None and j + 1 < tgt_ids.size(1):
                decoder_input = tgt_ids[:, j + 1]
            else:
                decoder_input = next_ids

        # Convert all_predict_words to tensor format expected by training script
        pred_tensor = torch.zeros(B, max_decoder_len, dtype=torch.long, device=device)
        for j in range(max_decoder_len):
            if j < len(all_predict_words):
                for b in range(B):
                    try:
                        pred_tensor[b, j] = int(all_predict_words[j][b])
                    except (ValueError, IndexError):
                        pred_tensor[b, j] = PAD_ID

        outputs = {
            "pred_ids": pred_tensor,
            "pointer_logits": all_pointer_logits,
            "vocab_logits": all_vocab_logits,
        }

        if tgt_ids is not None:
            outputs["loss"] = total_loss

        return outputs


# ----------------------------
# Example usage (toy)
# ----------------------------
if __name__ == "__main__":
    # toy vocab & mapping
    # Build a tiny vocabulary of characters mapped to integer ids
    vocab = [
        "<pad>",
        "这",
        "使",
        "我",
        "永",
        "生",
        "难",
        "望",
        "忘",
        "汪",
        "圣",
        "晚",
    ]  # V = 12
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    pad_idx = 0

    # create confusionset mapping: for each character id, which vocab ids are confusable
    # Example: confusion set of '望' includes '忘', '汪', '圣', '晚'
    conf_map: Dict[int, List[int]] = {
        stoi["望"]: [stoi["忘"], stoi["汪"], stoi["圣"], stoi["晚"]],
        stoi["难"]: [stoi["难"]],  # itself (no changes)
        # for characters not in map, we include itself as confusable
    }

    # function to build confusionset mask for a batch
    def build_confusion_mask(
        batch_src_ids: torch.Tensor, vocab_size: int, conf_map: Dict[int, List[int]]
    ):
        # batch_src_ids: (B, n)
        B, n = batch_src_ids.size()
        mask = torch.zeros((B, n, vocab_size), dtype=torch.uint8)
        for b in range(B):
            for i in range(n):
                cid = batch_src_ids[b, i].item()
                # include at least the char itself
                allowed = conf_map.get(cid, [cid])
                for a in allowed:
                    mask[b, i, a] = 1
        return mask

    # toy batch: "这 使 我 永 生 难 望"
    src_sentence = ["这", "使", "我", "永", "生", "难", "望"]
    src_ids = torch.tensor(
        [[stoi[c] for c in src_sentence]], dtype=torch.long
    )  # (1, 7)
    src_mask = (src_ids != pad_idx).long()

    # target sentence: correct '望' -> '忘'
    tgt_sentence = ["这", "使", "我", "永", "生", "难", "忘"]
    tgt_ids = torch.tensor([[stoi[c] for c in tgt_sentence]], dtype=torch.long)

    conf_mask = build_confusion_mask(src_ids, V, conf_map)  # (1, n, V)

    model = ConfusionPointerNet(
        vocab_size=23305,
        embed_dim=256,
        enc_hidden=256,
        dec_hidden=256,
        attn_dim=128,
    )

    model.load_pretrained_embeddings(
        "data/training/corpus/embeddings/cbow_decay/checkpoints/iter_epoch_100.pt"
    )

    outputs = model(src_ids, src_mask, conf_mask, tgt_ids=tgt_ids, teacher_forcing=True)
    print("Loss:", outputs["loss"].item())
    print("Pred ids:", outputs["pred_ids"].tolist())
    pred_chars = [[vocab[i] for i in row] for row in outputs["pred_ids"].tolist()]
    print("Pred chars:", pred_chars)
