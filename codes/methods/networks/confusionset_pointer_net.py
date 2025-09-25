import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..embeddings.cbow import CBOW

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
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)

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

        # vocab projection (from combined context)
        self.vocab_proj = nn.Linear(dec_hidden, vocab_size)

        # pointer network components
        # Wi and Wg in paper: make a small MLP that takes [Wg*Cj ; Locj]
        # We implement Wg as a linear on Cj
        self.Wg = nn.Linear(dec_hidden, dec_hidden)

        # Loc embedding isn't necessary as Loc is one-hot; instead we will concat one-hot Loc and Cj and project
        self.pointer_proj = nn.Linear(dec_hidden + 1, ptr_dim)
        self.pointer_score = nn.Linear(ptr_dim, 1)

        # optional dropout
        self.dropout = nn.Dropout(drop_rate)

        # sentinel handling is internal; no external token needed
        self.pad_idx = vocab_size
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
                    nn.init.xavier_uniform_(param)   # input -> hidden
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)       # hidden -> hidden
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # optional: set forget-gate bias to 1 (common trick for LSTMs)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
                
    def load_pretrained_embeddings(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained embeddings file not found at {path}")
        pretrained_weights = torch.load(path, map_location=torch.device('cpu'))
        cbow = CBOW(self.vocab_size, self.emb_dim)
        cbow.load_state_dict(pretrained_weights)
        logging.info("Pretrained embeddings loaded successfully.")
        self.embed.weight.data.copy_(cbow.in_embed.weight.data)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor):
        """
        src_ids: (B, n)
        src_mask: (B, n) 1 for real tokens, 0 for padding
        Returns:
            enc_hs: (B, n, 2*enc_hidden)
            enc_final: (h, c) if needed
        """
        emb = self.embed(src_ids)  # (B, n, E)
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
        tgt_ids: (B, n) target character ids (same length as input, because model fixes length)
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

        # initial input to decoder: typically <sos> or the first input char embedding; we follow paper structure where alignment is 1-1,
        # so we feed previous target token embedding (teacher forcing) or previous predicted.
        prev_input_ids = torch.full(
            (B,), self.pad_idx, dtype=torch.long, device=device
        )  # start with PAD token embedding
        results_pred = []
        pointer_logits_all = []
        vocab_logits_all = []
        loss_pointer = 0.0
        loss_gen = 0.0
        nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)  # we'll manage masking

        for j in range(n):
            # prepare input embedding
            emb_prev = self.embed(prev_input_ids)  # (B, E)
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
            vocab_logits_all.append(vocab_logits.unsqueeze(1))  # collect per timestep

            # --- Pointer logits (distribution over positions + sentinel) ---
            # create Locj: vector of length n with 1 at position j (paper sets j-th element 1 at timestep j)
            # For batch we create same Locj for all samples in batch (since decoding is position-synchronous)
            # Locj is (B, n) (some samples might have padding, but Locj always index j)
            Locj = torch.zeros(B, n, device=device)
            if j < n:
                Locj[:, j] = 1.0

            # The paper computes Lj = softmax(Wi[Wg Cj ; Locj]); we implement per-position scoring:
            # For each encoder position i, we create input vector concat([scalar Locj_i for i], Cj) and compute score
            # Instead we compute a score per encoder position by applying pointer_proj on each (Cj; Locj_i)
            # Build (B, n, 1) where Locj_i is scalar per position
            Locj_unsq = Locj.unsqueeze(-1)  # (B, n, 1)
            # expand combined (Cj) to (B, n, dec_hidden)
            Cj_exp = combined.unsqueeze(1).expand(-1, n, -1)
            pointer_input = torch.cat(
                [Cj_exp, Locj_unsq], dim=-1
            )  # (B, n, dec_hidden+1)
            pointer_hidden = torch.tanh(self.pointer_proj(pointer_input))  # (B, n, 128)
            pos_scores = self.pointer_score(pointer_hidden).squeeze(
                -1
            )  # (B, n) scores for each input position
            # sentinel score: produce an extra logit for sentinel using a small MLP on combined
            sentinel_logit = torch.tanh(self.Wg(combined)).sum(
                dim=-1, keepdim=True
            )  # (B, 1)
            pointer_logits = torch.cat([pos_scores, sentinel_logit], dim=-1)  # (B, n+1)
            # mask positions that are padding in src_mask: set to -inf
            pad_mask = torch.cat(
                [src_mask, torch.ones(B, 1, device=device)], dim=1
            )  # (B, n+1) allow sentinel
            pointer_logits = pointer_logits.masked_fill(pad_mask == 0, -1e9)
            pointer_logits_all.append(pointer_logits.unsqueeze(1))

            # --- Loss computation for training if target provided ---
            if tgt_ids is not None:
                # pointer location label Lloc_j:
                # if target at timestep j exists in input: label = index of its (first) match in input positions
                # otherwise label = n (index of sentinel) (0..n-1 are positions, n is sentinel)
                # We'll find for each sample the first position z such that src_ids[b, z] == tgt_ids[b,j]
                labels_pos = torch.full(
                    (B,), n, dtype=torch.long, device=device
                )  # default sentinel

                tgt_j = tgt_ids[:, j]  # (B,)

                # Only consider non-pad targets
                for b in range(B):
                    if tgt_j[b].item() == self.pad_idx:
                        # ignore pad targets (we'll set ignore index)
                        labels_pos[b] = -100
                        continue
                    # find first match in src_ids[b]
                    matches = (src_ids[b] == tgt_j[b]).nonzero(as_tuple=False)
                    if matches.numel() > 0:
                        labels_pos[b] = matches[0, 0].item()
                    else:
                        labels_pos[b] = n  # sentinel index

                # pointer loss (cross-entropy over n+1 classes)
                # pointer_logits: (B, n+1)
                valid_mask = labels_pos != -100
                if valid_mask.any():
                    pointer_loss_j = F.cross_entropy(
                        pointer_logits[valid_mask],
                        labels_pos[valid_mask],
                        reduction="sum",
                    )
                else:
                    pointer_loss_j = torch.tensor(0.0, device=device)
                loss_pointer = loss_pointer + pointer_loss_j

                # Generation loss: only for samples where label == sentinel (i.e., need to generate)
                generate_mask = labels_pos == n  # boolean (B,)
                if generate_mask.any():
                    # we must compute masked cross-entropy between vocab_logits and gold tgt_j for these samples,
                    # where allowed positions are confusionset_mask[b, j, :] (for timestep j)
                    # Build logits and mask for only the generating samples
                    gen_logits = vocab_logits[generate_mask]  # (G, V)
                    # corresponding masks:
                    gen_mask = confusionset_mask[generate_mask, j, :]  # (G, V)
                    # If for some sample the target is not in mask (shouldn't happen if confusionset covers true char),
                    # we still allow full vocab as fallback. We'll check and adjust:
                    tgt_gen = tgt_j[generate_mask]  # (G,)
                    # Check if target is in mask; if not, expand mask to include target
                    # (rare but safe)
                    for idx in range(gen_mask.size(0)):
                        if gen_mask[idx, tgt_gen[idx]] == 0:
                            gen_mask[idx, tgt_gen[idx]] = 1
                    # compute masked probabilities with masked_softmax on logits
                    # For loss use cross-entropy: we compute logits masked by -inf at disallowed positions and then CE
                    neg_inf = -1e9
                    masked_logits = gen_logits.masked_fill(
                        gen_mask == 0, neg_inf
                    )  # (G, V)
                    # if some row has all zeros mask (shouldn't), fallback to unmasked logits
                    # compute CE
                    gen_loss_j = F.cross_entropy(
                        masked_logits, tgt_gen, reduction="sum"
                    )
                else:
                    gen_loss_j = torch.tensor(0.0, device=device)
                loss_gen = loss_gen + gen_loss_j

            # --- Greedy prediction for next step (and for inference) ---
            # pointer decision: if argmax pointer_logits != sentinel -> copy; else generate from masked vocab (apply mask then argmax)
            with torch.no_grad():
                ptr_choice = pointer_logits.argmax(dim=-1)  # (B,)
                next_ids = torch.full(
                    (B,), self.pad_idx, dtype=torch.long, device=device
                )
                # copy where ptr_choice < n
                copy_mask = ptr_choice < n
                if copy_mask.any():
                    idxs = copy_mask.nonzero(as_tuple=False).squeeze(-1)
                    for b in idxs:
                        pos = ptr_choice[b].item()
                        next_ids[b] = src_ids[b, pos].item()
                # generate where ptr_choice == n (sentinel)
                gen_mask_bool = ptr_choice == n
                if gen_mask_bool.any():
                    gen_idxs = gen_mask_bool.nonzero(as_tuple=False).squeeze(-1)
                    # compute masked probs and pick argmax
                    gen_logits_all = vocab_logits[gen_idxs]  # (G, V)
                    masks = confusionset_mask[gen_idxs, j, :]  # (G, V)
                    # fallback if mask sums to zero -> allow full vocab
                    sums = masks.sum(dim=-1)
                    for t_i in range(masks.size(0)):
                        if sums[t_i] == 0:
                            masks[t_i] = 1
                    masked_probs = masked_softmax(gen_logits_all, masks, dim=-1)
                    chosen = masked_probs.argmax(dim=-1)
                    for k_i, b in enumerate(gen_idxs):
                        next_ids[b] = chosen[k_i].item()

            # append predicted id
            results_pred.append(next_ids.unsqueeze(1))
            prev_input_ids = next_ids  # feed as prev input in next timestep

        # stack results
        pred_tensor = torch.cat(results_pred, dim=1)  # (B, n)
        pointer_logits_tensor = torch.cat(pointer_logits_all, dim=1)  # (B, n, n+1)
        vocab_logits_tensor = torch.cat(vocab_logits_all, dim=1)  # (B, n, V)

        outputs = {
            "pred_ids": pred_tensor,
            "pointer_logits": pointer_logits_tensor,
            "vocab_logits": vocab_logits_tensor,
        }

        if tgt_ids is not None:
            # total loss normalized by batch size (or by number of tokens) - follow your preference
            total_loss = (loss_pointer + loss_gen) / src_ids.size(0)
            outputs["loss"] = total_loss
            outputs["loss_pointer_sum"] = loss_pointer.detach()
            outputs["loss_gen_sum"] = loss_gen.detach()

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
    
    model.load_pretrained_embeddings("data/training/corpus/embeddings/cbow_decay/checkpoints/iter_epoch_100.pt")

    outputs = model(src_ids, src_mask, conf_mask, tgt_ids=tgt_ids, teacher_forcing=True)
    print("Loss:", outputs["loss"].item())
    print("Pred ids:", outputs["pred_ids"].tolist())
    pred_chars = [[vocab[i] for i in row] for row in outputs["pred_ids"].tolist()]
    print("Pred chars:", pred_chars)
