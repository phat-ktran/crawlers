import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RadicalEnhancedCharEmbedding(nn.Module):
    def __init__(self, char_vocab_size, radical_vocab_size, embed_dim=30, window_size=5, hidden_dim=30):
        super(RadicalEnhancedCharEmbedding, self).__init__()
        
        self.char_vocab_size = char_vocab_size
        self.radical_vocab_size = radical_vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.char_embeddings = nn.Embedding(char_vocab_size, embed_dim)
        # Note: Radicals are not embedded as input; they are prediction targets.
        
        # Shared hidden layer for context task (HardTanh)
        self.context_hidden = nn.Linear(window_size * embed_dim, hidden_dim)
        
        # Context task: score the ngram (scalar output)
        self.context_score = nn.Linear(hidden_dim, 1)
        
        # Radical task: for each character in window, predict its radical
        # Shared across all positions in window
        self.radical_predictor = nn.Linear(embed_dim, radical_vocab_size)
        
        # Activation
        self.hardtanh = nn.Hardtanh()  # As used in paper

    def forward_context(self, char_ids):
        """
        Forward pass for context task.
        char_ids: Tensor of shape (batch_size, window_size) - character indices in window
        Returns: scalar score for each ngram (batch_size,)
        """
        batch_size = char_ids.size(0)
        # Lookup embeddings: (batch_size, window_size, embed_dim)
        embeds = self.char_embeddings(char_ids)
        # Flatten: (batch_size, window_size * embed_dim)
        embeds_flat = embeds.view(batch_size, -1)
        # Hidden layer
        hidden = self.hardtanh(self.context_hidden(embeds_flat))
        # Score
        score = self.context_score(hidden).squeeze(-1)  # (batch_size,)
        return score

    def forward_radical(self, char_ids):
        """
        Forward pass for radical prediction task.
        char_ids: Tensor of shape (batch_size, window_size)
        Returns: logits for radical prediction, shape (batch_size, window_size, radical_vocab_size)
        """
        batch_size = char_ids.size(0)
        # Lookup embeddings: (batch_size, window_size, embed_dim)
        embeds = self.char_embeddings(char_ids)
        # For each position in window, predict radical
        # Reshape to (batch_size * window_size, embed_dim)
        embeds_reshaped = embeds.view(-1, self.embed_dim)
        # Predict radicals: (batch_size * window_size, radical_vocab_size)
        radical_logits_flat = self.radical_predictor(embeds_reshaped)
        # Reshape back: (batch_size, window_size, radical_vocab_size)
        radical_logits = radical_logits_flat.view(batch_size, self.window_size, self.radical_vocab_size)
        return radical_logits

    def forward(self, char_ids):
        """
        Combined forward for both tasks.
        char_ids: (batch_size, window_size)
        Returns:
            context_score: (batch_size,)
            radical_logits: (batch_size, window_size, radical_vocab_size)
        """
        context_score = self.forward_context(char_ids)
        radical_logits = self.forward_radical(char_ids)
        return context_score, radical_logits


class RadicalEnhancedTrainer:
    def __init__(self, model, char_to_radical, alpha=0.5, lr=0.1):
        self.model = model
        self.char_to_radical = char_to_radical  # dict: char_id -> radical_id
        self.alpha = alpha
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.window_size = model.window_size

    def generate_training_batch(self, char_corpus, batch_size=32):
        """
        Generate one batch for training.
        char_corpus: list of character indices (whole corpus)
        Returns:
            real_batch: (batch_size, window_size) - real ngrams
            corrupt_batch: (batch_size, window_size) - corrupted ngrams (center char replaced)
            radical_labels_real: (batch_size, window_size) - radical IDs for real batch
            radical_labels_corrupt: (batch_size, window_size) - for corrupt batch
        """
        real_ngrams = []
        corrupt_ngrams = []
        rad_labels_real = []
        rad_labels_corrupt = []

        half_win = self.window_size // 2
        corpus_len = len(char_corpus)

        for _ in range(batch_size):
            # Sample random center position
            center_idx = random.randint(half_win, corpus_len - half_win - 1)
            # Extract real ngram
            ngram = char_corpus[center_idx - half_win: center_idx + half_win + 1]
            real_ngrams.append(ngram)
            # Create corrupt ngram: replace center char with random char
            corrupt_ngram = ngram.copy()
            random_char = random.randint(0, self.model.char_vocab_size - 1)
            corrupt_ngram[half_win] = random_char
            corrupt_ngrams.append(corrupt_ngram)

            # Get radical labels for real and corrupt ngrams
            rad_real = [self.char_to_radical.get(c, 0) for c in ngram]  # default 0 if not found
            rad_corrupt = [self.char_to_radical.get(c, 0) for c in corrupt_ngram]
            rad_labels_real.append(rad_real)
            rad_labels_corrupt.append(rad_corrupt)

        return (torch.tensor(real_ngrams),
                torch.tensor(corrupt_ngrams),
                torch.tensor(rad_labels_real),
                torch.tensor(rad_labels_corrupt))

    def train_step(self, real_batch, corrupt_batch, rad_real, rad_corrupt):
        self.optimizer.zero_grad()

        # Forward pass for real ngrams
        score_real, rad_logits_real = self.model(real_batch)
        # Forward pass for corrupt ngrams
        score_corrupt, rad_logits_corrupt = self.model(corrupt_batch)

        # Context Loss (C&W ranking loss)
        # loss = max(0, 1 - score(real) + score(corrupt))
        context_loss = F.relu(1.0 - score_real + score_corrupt).mean()

        # Radical Loss (Cross-Entropy)
        # Flatten for loss computation
        rad_logits_real_flat = rad_logits_real.view(-1, self.model.radical_vocab_size)
        rad_real_flat = rad_real.view(-1)
        rad_loss_real = F.cross_entropy(rad_logits_real_flat, rad_real_flat)

        rad_logits_corrupt_flat = rad_logits_corrupt.view(-1, self.model.radical_vocab_size)
        rad_corrupt_flat = rad_corrupt.view(-1)
        rad_loss_corrupt = F.cross_entropy(rad_logits_corrupt_flat, rad_corrupt_flat)

        radical_loss = rad_loss_real + rad_loss_corrupt

        # Total Loss
        total_loss = self.alpha * context_loss + (1 - self.alpha) * radical_loss

        # Backward
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), context_loss.item(), radical_loss.item()


# ======== USAGE EXAMPLE ========

if __name__ == "__main__":
    # Example: Tiny toy corpus and mappings
    # In practice, load from real corpus and radical dictionary

    # Suppose we have 100 unique characters and 10 radicals
    CHAR_VOCAB_SIZE = 100
    RADICAL_VOCAB_SIZE = 10
    EMBED_DIM = 30
    WINDOW_SIZE = 5

    # Dummy char-to-radical mapping (in real case, load from dictionary)
    char_to_radical = {i: i % RADICAL_VOCAB_SIZE for i in range(CHAR_VOCAB_SIZE)}

    # Dummy corpus: list of character indices
    corpus = list(range(CHAR_VOCAB_SIZE)) * 100  # repeat for training

    # Initialize model and trainer
    model = RadicalEnhancedCharEmbedding(
        char_vocab_size=CHAR_VOCAB_SIZE,
        radical_vocab_size=RADICAL_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        window_size=WINDOW_SIZE
    )

    trainer = RadicalEnhancedTrainer(model, char_to_radical, alpha=0.5, lr=0.1)

    # Training loop
    for epoch in range(500):
        real_b, corrupt_b, rad_real_b, rad_corrupt_b = trainer.generate_training_batch(corpus, batch_size=32)
        total_loss, ctx_loss, rad_loss = trainer.train_step(real_b, corrupt_b, rad_real_b, rad_corrupt_b)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss:.4f} | Context: {ctx_loss:.4f} | Radical: {rad_loss:.4f}")

    # After training, extract character embeddings
    char_embeddings = model.char_embeddings.weight.data  # Shape: (char_vocab_size, embed_dim)
    print("Character Embeddings Shape:", char_embeddings.shape)
    print("First 5 character embeddings:\n", char_embeddings[:5])