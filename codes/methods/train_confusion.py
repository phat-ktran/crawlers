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


def build_confusion_mask(
    batch_src_ids: torch.Tensor, vocab_size: int, conf_map: Dict[int, List[int]]
):
    B, n = batch_src_ids.size()
    mask = torch.zeros((B, n, vocab_size), dtype=torch.uint8)
    for b in range(B):
        for i in range(n):
            cid = batch_src_ids[b, i].item()
            allowed = conf_map.get(cid, [])
            for a in allowed:
                mask[b, i, a] = 1
    return mask


class CorrectionDataset(Dataset):
    def __init__(self, data, stoi):
        self.samples = []
        self.pad_idx = len(stoi)
        for error_seq, gt_seq, viet_seq in data:
            src_ids = [stoi.get(c, self.pad_idx) for c in error_seq]
            tgt_ids = [stoi.get(c, self.pad_idx) for c in gt_seq]
            self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, V, conf_map):
    srcs, tgts = zip(*batch)
    padded_src = pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in srcs],
        batch_first=True,
        padding_value=V,
    )
    padded_tgt = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tgts],
        batch_first=True,
        padding_value=V,
    )
    src_mask = (padded_src != V).long()
    conf_mask = build_confusion_mask(padded_src, V, conf_map)
    return padded_src, src_mask, conf_mask, padded_tgt


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
                viet_list = list(viet_seq.strip())
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
        "--confusion",
        default="codes/assets/similarity.csv",
        help="CSV file for confusion network (sep ':').",
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
    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    logging.info(f"Vocabulary size: {V}")

    # Load confusion map
    conf_map: Dict[int, List[int]] = {}
    with open(args.confusion, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=":")
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 2:
                char = row[0].strip()
                conf_str = row[1].strip()
                confs = list(conf_str)
                if char in stoi:
                    conf_ids = [stoi[c] for c in confs if c in stoi]
                    conf_map[stoi[char]] = conf_ids

    # Add self to all confusion sets
    for i in range(V):
        if i not in conf_map:
            conf_map[i] = [i]
        elif i not in conf_map[i]:
            conf_map[i].append(i)
    logging.info(f"Loaded confusion map with {len(conf_map)} entries.")

    # Load datasets
    train_data = load_data(args.train)
    val_data = load_data(args.val)
    logging.info(
        f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples."
    )

    # Datasets and DataLoaders
    train_ds = CorrectionDataset(train_data, stoi)
    val_ds = CorrectionDataset(val_data, stoi)
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
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Using device: {device}")

    # Training loop
    total_steps = len(train_loader)
    logging.info(f"Total steps per epoch: {total_steps}")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            src, src_mask, conf_mask, tgt = [x.to(device) for x in batch]
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
                    pred_list = pred_ids[b][pred_ids[b] != V].tolist()
                    tgt_list = tgt[b][tgt[b] != V].tolist()
                    dist = levenshtein_distance(pred_list, tgt_list)
                    cer = dist / len(tgt_list) if len(tgt_list) > 0 else 0.0
                    cer_list.append(cer)
                avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0
                
                logging.info(f"Epoch {epoch + 1}, Step {step}: Loss = {loss.item()}, CER = {avg_cer:0.4f}")

        logging.info(f"Completed epoch {epoch + 1}/{args.epochs}, moving to validation.")

        # Validation
        model.eval()
        with torch.no_grad():
            cer_list = []
            for batch in val_loader:
                src, src_mask, conf_mask, tgt = [x.to(device) for x in batch]
                outputs = model(
                    src, src_mask, conf_mask, tgt_ids=tgt, teacher_forcing=False
                )
                pred_ids = outputs["pred_ids"]  # (B, seq_len)
                B = pred_ids.size(0)
                for b in range(B):
                    pred_list = pred_ids[b][pred_ids[b] != V].tolist()
                    tgt_list = tgt[b][tgt[b] != V].tolist()
                    dist = levenshtein_distance(pred_list, tgt_list)
                    cer = dist / len(tgt_list) if len(tgt_list) > 0 else 0.0
                    cer_list.append(cer)
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
