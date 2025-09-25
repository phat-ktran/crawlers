import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from collections import Counter
import logging

class SkipGramDataset(Dataset):
    def __init__(self, tokenized_corpus, window_size=5):
        self.pairs = []
        for sent in tokenized_corpus:
            for i in range(len(sent)):
                for j in range(max(0, i - window_size), min(len(sent), i + window_size + 1)):
                    if i != j:
                        self.pairs.append((sent[i], sent[j]))  # target, context

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGramNS, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.xavier_uniform_(self.out_embed.weight)

    def forward(self, target, context, noise):
        emb_target = self.in_embed(target)  # batch_size x emb_dim
        emb_context = self.out_embed(context)  # batch_size x emb_dim
        emb_noise = self.out_embed(noise)  # batch_size x num_neg x emb_dim

        # Positive loss
        pos_dot = torch.sum(emb_target * emb_context, dim=1)  # batch_size
        pos_loss = -torch.log(torch.sigmoid(pos_dot))  # batch_size

        # Negative loss
        neg_dot = torch.bmm(emb_noise, emb_target.unsqueeze(2)).squeeze(2)  # batch_size x num_neg
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_dot)), dim=1)  # batch_size

        return (pos_loss + neg_loss).mean()

def main():
    parser = argparse.ArgumentParser(description="Train SkipGram model with Negative Sampling in PyTorch.")
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
    args = parser.parse_args()

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
    dataset = SkipGramDataset(tokenized_corpus, window_size=args.window_size)
    logging.info(f"Dataset size: {len(dataset)} pairs.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model, optimizer, device
    emb_dim = args.emb_dim
    num_neg = args.num_neg
    model = SkipGramNS(vocab_size, emb_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    noise_dist = noise_dist.to(device)
    logging.info(f"Using device: {device}")

    # Training loop
    losses_file = os.path.join(args.output_dir, 'losses.txt')
    with open(losses_file, 'w') as f:
        f.write("step,loss\n")

    step = 0
    steps_per_epoch = len(dataloader)
    logging.info(f"Number of steps per epoch: {steps_per_epoch}")
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        for batch in dataloader:
            target, context = batch
            target = target.to(device)
            context = context.to(device)

            # Sample negatives
            noise = torch.multinomial(noise_dist, target.size(0) * num_neg, replacement=True)
            noise = noise.view(target.size(0), num_neg).to(device)

            optimizer.zero_grad()
            loss = model(target, context, noise)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            step += 1
            if step % 100 == 0:
                current_loss = loss.item()
                logging.info(f"Step {step}: Loss = {current_loss}")
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