import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
from collections import Counter
import logging
import torch.nn.functional as F


class CBOWDataset(Dataset):
    def __init__(self, tokenized_corpus, window_size=5):
        self.samples = []
        for sent in tokenized_corpus:
            for i in range(len(sent)):
                context = [sent[j] for j in range(max(0, i - window_size), min(len(sent), i + window_size + 1)) if j != i]
                if context:
                    self.samples.append((context, sent[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context), torch.tensor(target)

def collate_fn(batch):
    contexts, targets = zip(*batch)
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=vocab_size)
    targets = torch.stack(targets)
    return padded_contexts, targets

class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(CBOW, self).__init__()
        self.in_embed = nn.Embedding(vocab_size+1, emb_dim, padding_idx=vocab_size)
        self.out_embed = nn.Embedding(vocab_size+1, emb_dim, padding_idx=vocab_size)
        initrange = 0.5 / emb_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()
        self.in_embed.weight.data[vocab_size] = 0
        self.out_embed.weight.data[vocab_size] = 0

    def forward(self, contexts, target, noise):
        emb_contexts = self.in_embed(contexts)  # batch_size x max_context x emb_dim
        num_contexts = (contexts != vocab_size).sum(dim=1, keepdim=True).float().clamp(min=1)
        avg_emb = emb_contexts.sum(dim=1) / num_contexts  # batch_size x emb_dim

        emb_target = self.out_embed(target)  # batch_size x emb_dim
        emb_noise = self.out_embed(noise)  # batch_size x num_neg x emb_dim

        # Positive loss
        pos_dot = torch.sum(avg_emb * emb_target, dim=1)  # batch_size
        pos_loss = -F.logsigmoid(pos_dot)  # batch_size

        # Negative loss
        neg_dot = torch.bmm(emb_noise, avg_emb.unsqueeze(2)).squeeze(2)  # batch_size x num_neg
        neg_loss = -torch.sum(F.logsigmoid(-neg_dot), dim=1)  # batch_size

        return (pos_loss + neg_loss).mean()
        
def main():
    parser = argparse.ArgumentParser(description="Train CBOW model with Negative Sampling in PyTorch.")
    parser.add_argument('--input', required=True, help="Path to the input text corpus (each line is a text sequence in Chinese).")
    parser.add_argument('--vocab', required=True, help="Path to the vocabulary file (each line is a character).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--output-dir', required=True, help="Directory to save model weights and logs.")
    parser.add_argument('--window-size', type=int, default=5, help="Size of the context window for SkipGram.")
    parser.add_argument('--batch-size', type=int, default=512, help="Batch size for DataLoader.")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker threads for DataLoader.")
    parser.add_argument('--num-neg', type=int, default=5, help="Number of negative samples for each positive pair.")
    parser.add_argument('--emb-dim', type=int, default=512, help="Dimension of the embedding vectors.")
    parser.add_argument('--optim', type=str, default="Adam", help="Optimizer to use.")
    args = parser.parse_args()

    global vocab_size
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting training process.")

    # Read vocabulary
    with open(args.vocab, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f if line.strip()]
    word2idx = {char: idx for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    # Read corpus
    corpus = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            seq = line.strip()
            if seq:
                corpus.append(seq)
    logging.info(f"Loaded {len(corpus)} sequences from corpus.")

    # Tokenize corpus
    tokenized_corpus = [[word2idx[c] for c in seq if c in word2idx] for seq in corpus]

    # Compute frequency for negative sampling distribution
    all_tokens = [token for sent in tokenized_corpus for token in sent]
    freq = Counter(all_tokens)
    noise_dist = torch.tensor([freq.get(i, 0) ** 0.75 for i in range(vocab_size)]).float()
    if noise_dist.sum() == 0:
        noise_dist = torch.ones(vocab_size) / vocab_size
    else:
        noise_dist /= noise_dist.sum()

    # Dataset and DataLoader
    dataset = CBOWDataset(tokenized_corpus, window_size=5)
    logging.info(f"Dataset size: {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    # Model, optimizer, device
    emb_dim = args.emb_dim
    num_neg = args.num_neg
    model = CBOW(vocab_size, emb_dim)
    
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optim == "Adam" else optim.SGD(model.parameters(), lr=args.lr)
    lr_min = 1e-4 
    
    def lr_lambda(current_step):
        progress = float(current_step) / float(total_steps)
        # Linearly decay from 1.0 -> 0.0, then scale into [lr_min, lr_start]
        return (1 - progress) * (args.lr - lr_min) / args.lr + lr_min / args.lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    noise_dist = noise_dist.to(device)
    logging.info(f"Using device: {device}")

    # Losses file
    losses_file = os.path.join(args.output_dir, 'losses.txt')
    with open(losses_file, 'w') as f:
        f.write("step,loss\n")

    step = 0
    
    logging.info(f"Number of steps per epoch: {steps_per_epoch}")
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        for batch in dataloader:
            contexts, target = batch
            contexts = contexts.to(device)
            target = target.to(device)

            # Sample negatives
            noise = torch.multinomial(noise_dist, target.size(0) * num_neg, replacement=True)
            noise = noise.view(target.size(0), num_neg).to(device)

            optimizer.zero_grad()
            loss = model(contexts, target, noise)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_loss = loss.item()
                logging.info(f"Step {step}: Loss = {current_loss}, lr={current_lr:.6f},")
                with open(losses_file, 'a') as f:
                    f.write(f"{step},{current_loss}\n")

        logging.info(f"Completed epoch {epoch + 1}/{args.epochs}")

        # Save checkpoint for the current epoch
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, "checkpoints", f'iter_epoch_{epoch + 1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    main()