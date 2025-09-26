import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import logging
from typing import Dict, List
import csv

from codes.methods.networks.confusionset_pointer_net import ConfusionPointerNet
from codes.methods.utils import (
    SOS_ID,
    PAD_ID,
    EOS_ID,
    UNK_ID,
)

def token2str(ids, mask, id2token, training=False):
    """
    Convert list of IDs to string, ignoring special tokens and truncating at EOS.
    The mask and training parameters are retained for compatibility but unused.
    """
    filtered_ids = []
    for i in ids:
        if i == EOS_ID:
            break
        if i not in {SOS_ID, PAD_ID, UNK_ID}:
            filtered_ids.append(i)
    return ''.join(id2token.get(i, '<unk>') for i in filtered_ids)

def build_confusion_mask(
    batch_src_ids: torch.Tensor,
    batch_viet_seqs: List[List[str]],
    vocab_size: int,
    conf_map: Dict[str, List[int]],
):
    """
    batch_src_ids: (B, n) tensor of Sino-Nôm token IDs
    batch_viet_seqs: list of Vietnamese tokenized sequences, len = B
    vocab_size: size of vocab
    conf_map: dictionary {viet_word: [candidate_sino_ids]}
    """
    B, n = batch_src_ids.size()
    mask = torch.zeros((B, n, vocab_size), dtype=torch.uint8)

    for b in range(B):
        viet_words = batch_viet_seqs[b]  # list of tokens
        for i in range(n):
            cid = batch_src_ids[b, i].item()
            if cid in {PAD_ID, SOS_ID, EOS_ID, UNK_ID}:
                continue

            # If we have a Vietnamese word aligned at position i
            if i < len(viet_words):
                viet_word = viet_words[i]
                allowed = conf_map.get(viet_word, [])
            else:
                allowed = []

            # Always allow the original char id (so identity is possible)
            allowed = set(allowed) | {cid}

            for a in allowed:
                mask[b, i, a] = 1
    return mask


class CorrectionDataset(Dataset):
    def __init__(self, data, stoi):
        """
        data: list of (error_seq, gt_seq, viet_seq)
        stoi: vocab mapping {char: id}
        """
        self.samples = []
        for error_seq, gt_seq, viet_seq in data:
            src_ids = [stoi.get(c, UNK_ID) for c in error_seq]
            gt_ids = [stoi.get(c, UNK_ID) for c in gt_seq]

            # add <SOS> and <EOS> for the target
            tgt_ids = [SOS_ID] + gt_ids + [EOS_ID]

            self.samples.append((src_ids, tgt_ids, viet_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, V, conf_map):
    """
    batch: list of (src_ids, tgt_ids, viet_seq)
    V: vocab size
    conf_map: confusion mapping (viet_word -> sino_candidates)
    """
    srcs, tgts, viet_seqs = zip(*batch)

    # Pad source
    padded_src = pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in srcs],
        batch_first=True,
        padding_value=PAD_ID,
    )

    # Pad target (with <SOS>, <EOS>)
    padded_tgt = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tgts],
        batch_first=True,
        padding_value=PAD_ID,
    )

    # Build shifted target for teacher forcing
    y_shift = padded_tgt[:, :-1]
    y = padded_tgt[:, 1:]

    # Source mask
    src_mask = (padded_src != PAD_ID).long()
    

    # Confusion mask (use viet seqs)
    conf_mask = build_confusion_mask(padded_src, viet_seqs, V, conf_map)

    return padded_src, src_mask, conf_mask, y_shift, y, list(viet_seqs)

def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                error_seq, gt_seq, _, viet_seq = parts
                # Treat sequences as lists of characters
                error_list = list(error_seq.strip())
                gt_list = list(gt_seq.strip())
                viet_list = viet_seq.strip().split()
                data.append((error_list, gt_list, viet_list))
    return data


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def main():
    parser = argparse.ArgumentParser(description="Train ConfusionPointerNet model.")
    parser.add_argument("--train", required=True, help="Path to training dataset.")
    parser.add_argument("--val", required=True, help="Path to validation dataset.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--vocab",
        default="data/training/corpus/labels/vocab.txt",
        help="Path to vocabulary file (each line a character).",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save logs and checkpoints."
    )
    parser.add_argument(
        "--dict",
        default="codes/assets/dict.csv",
        help="CSV file for dict (sep ':').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--pretrained-embs",
        type=str,
        default=None,
        help="Pretrained weights for character embeddings.",
    )
    parser.add_argument("--save_res_path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training process.")

    # Load vocabulary
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<sos>": SOS_ID, "<eos>": EOS_ID}
    with open(args.vocab, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vocab[line.strip()] = len(vocab)
    V = len(vocab)
    logging.info(f"Vocabulary size: {V}")
    
    id2token = {v: k for k, v in vocab.items()}

    # Load confusion map
    conf_map: Dict[str, List[int]] = {}
    with open(args.dict, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 2:
                vn = row[0].strip()
                nom = row[1].strip()
                if nom not in vocab:
                    continue
                if vn not in conf_map:
                    conf_map[vn] = []
                conf_map[vn].append(vocab[nom])
    logging.info(f"Loaded dictionary with {len(conf_map)} entries.")

    # Load datasets
    train_data = load_data(args.train)
    val_data = load_data(args.val)
    logging.info(
        f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples."
    )

    # Datasets and DataLoaders
    train_ds = CorrectionDataset(train_data, vocab)
    val_ds = CorrectionDataset(val_data, vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, V, conf_map),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, V, conf_map),
    )

    # Model, optimizer, device
    model = ConfusionPointerNet(
        vocab_size=V,
        embed_dim=256,
        enc_hidden=384,
        dec_hidden=384,
        attn_dim=384,
        drop_rate=0.5
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.pretrained_embs:
        model.load_pretrained_embeddings(args.pretrained_embs)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    logging.info(f"Using device: {device}")

    # Training loop
    total_steps = len(train_loader)
    logging.info(f"Total steps per epoch: {total_steps}")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            src, src_mask, conf_mask, tgt, y = [x.to(device) for x in batch[:5]]
            optimizer.zero_grad()
            outputs = model(src, src_mask, conf_mask, tgt_ids=tgt, teacher_forcing=True)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if step % 10 == 0:
                cer_list = []
                pred_ids = outputs["pred_ids"]  # (B, seq_len)
                B = pred_ids.size(0)
                for b in range(B):
                    pred_list_full = pred_ids[b][pred_ids[b] != PAD_ID].tolist()
                    y_list_full = y[b][y[b] != PAD_ID].tolist()
                    
                    # Truncate at first EOS if present
                    try:
                        eos_idx_pred = pred_list_full.index(EOS_ID)
                        pred_list = pred_list_full[:eos_idx_pred]
                    except ValueError:
                        pred_list = pred_list_full  # No EOS, use all
                    
                    try:
                        eos_idx_y = y_list_full.index(EOS_ID)
                        gt_list = y_list_full[:eos_idx_y]
                    except ValueError:
                        gt_list = y_list_full
                    
                    dist = levenshtein_distance(pred_list, gt_list)
                    cer = dist / len(gt_list) if len(gt_list) > 0 else 0.0
                    cer_list.append(cer)
                
                if f:
                    f.close()
                avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0

                logging.info(f"Epoch {epoch + 1}, Step {step}: Loss = {loss.item()}, CER = {avg_cer:0.4f}")

        logging.info(f"Completed epoch {epoch + 1}/{args.epochs}, moving to validation.")

        # Validation
        model.eval()
        with torch.no_grad():
            f = None
            if args.save_res_path is not None:
                f = open(args.save_res_path, "a")
                f.write(f"\n======EPOCH {epoch+1}/{args.epochs}======\n")
                
            cer_list = []
            for batch in val_loader:
                src, src_mask, conf_mask, tgt, y = [x.to(device) for x in batch[:5]]
                outputs = model(
                    src, src_mask, conf_mask, tgt_ids=tgt, teacher_forcing=False
                )
                pred_ids = outputs["pred_ids"]  # (B, seq_len)
                B = pred_ids.size(0)
                y_cpu = y.to("cpu")
                for b in range(B):
                    pred_list_full = pred_ids[b][pred_ids[b] != PAD_ID].tolist()
                    y_list_full = y[b][y[b] != PAD_ID].tolist()
                    
                    # Truncate at first EOS if present
                    try:
                        eos_idx_pred = pred_list_full.index(EOS_ID)
                        pred_list = pred_list_full[:eos_idx_pred]
                    except ValueError:
                        pred_list = pred_list_full  # No EOS, use all
                    
                    try:
                        eos_idx_y = y_list_full.index(EOS_ID)
                        gt_list = y_list_full[:eos_idx_y]
                    except ValueError:
                        gt_list = y_list_full
                    
                    dist = levenshtein_distance(pred_list, gt_list)                    
                    pred = token2str(pred_list, None, id2token, training=False)
                    truth = token2str(gt_list, None, id2token, training=False)
                    
                    if f is not None:
                        f.write(f"Prediction: {pred}\tGround Truth: {truth}\n")
                    
                    cer = dist / len(gt_list) if len(gt_list) > 0 else 0.0
                    cer_list.append(cer)
            if f is not None:
                f.close()
            avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0
        logging.info(f"Epoch {epoch + 1}/{args.epochs}: CER = {avg_cer}")

        # Save checkpoint
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        checkpoint_path = os.path.join(
            args.output_dir, "checkpoints", f"iter_epoch_{epoch + 1}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")

    logging.info("Training completed.")


if __name__ == "__main__":
    main()