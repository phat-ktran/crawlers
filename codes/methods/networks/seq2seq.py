import argparse
import os
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from codes.methods.networks.pmam.attn import TransformBlock
from codes.methods.embeddings.cbow import CBOW
from codes.methods.utils import SOS_ID, UNK_ID, PAD_ID, EOS_ID

class SinoNomDataset(Dataset):
    def __init__(self, lines, vocab, bypass_check=False):
        self.data = []
        for line in lines:
            parts = line.strip().split("\t")
            x = [vocab.get(c, UNK_ID) for c in parts[0].strip()]
            y = [vocab.get(c, UNK_ID) for c in parts[1].strip()]

            # 🔑 Append EOS to target
            y.append(EOS_ID)

            b = [int(bb) for bb in parts[2].split(",")]
            t_text = parts[3]

            if not bypass_check:
                assert len(x) == len(y) - 1 == len(b) == len(t_text.split()), \
                    f"Data mismatch: x({parts[0]}), y({parts[1]}), b({parts[2]}), t_text({parts[3]})"

            self.data.append((x, y, b, t_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    xs, ys, bs, ts = zip(*batch)
    lens = [len(y) for y in ys]   # use y length (includes EOS)
    max_l = max(lens)

    pad_x = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    pad_y = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    pad_y_shift = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    pad_b = torch.zeros((len(batch), max_l), dtype=torch.float)
    mask = torch.zeros((len(batch), max_l), dtype=torch.float)

    for i in range(len(batch)):
        l = lens[i]
        pad_x[i, :len(xs[i])] = torch.tensor(xs[i])
        pad_y[i, :l] = torch.tensor(ys[i])
        pad_b[i, :len(bs[i])] = torch.tensor(bs[i], dtype=torch.float)

        mask[i, :l] = 1  # EOS is included in mask

        # Teacher forcing input: SOS + shifted y (without last EOS)
        pad_y_shift[i, 0] = SOS_ID
        pad_y_shift[i, 1:l] = torch.tensor(ys[i][:-1])

    return pad_x, pad_y_shift, pad_y, pad_b, list(ts), mask



class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        transform: TransformBlock,
        embed_dim=300,
        hidden_size=256,
        att_dim=256,
        drop_rate=0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout_e = nn.Dropout(p=drop_rate)
        self.sino_encoder = nn.LSTM(
            embed_dim, hidden_size, 1, bidirectional=True, batch_first=True
        )
        self.transform = transform
        self.U = nn.Linear(hidden_size, att_dim)
        self.v = nn.Linear(att_dim, 1)
        self.dropout_u = nn.Dropout(p=drop_rate)
        self.dropout_v = nn.Dropout(p=drop_rate)
        self.decoder_lstm = nn.LSTMCell(embed_dim, hidden_size)
        self.output_linear = nn.Linear(
            self.transform.hidden_size + hidden_size,
            vocab_size,
        )
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                
            elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
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
            elif isinstance(m, TransformBlock):
                m.init_weights()

    def load_pretrained_embeddings(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained embeddings file not found at {path}")
        pretrained_weights = torch.load(path, map_location=torch.device('cpu'))
        cbow = CBOW(self.vocab_size - 4, self.emb_dim)
        cbow.load_state_dict(pretrained_weights)
        self.embedding.weight.data[-(self.vocab_size - 4):, :].copy_(cbow.in_embed.weight.data[:-1, :])

    def forward(self, x, viet_texts, y_shift=None, mask=None):
        device = x.device

        embed_x = self.embedding(x)  # shape: (batch_size, seq_len, embed_dim)
        embed_x = self.dropout_e(embed_x)
        e, _ = self.sino_encoder(
            embed_x
        )  # e shape: (batch_size, seq_len, 2 * hidden_size)
        # tilde_h shape: (batch_size, seq_len, att_dim)
        # raw_enc shape: (batch_size, seq_len, self.transform.hidden_dim)
        raw_enc, tilde_h, l = self.transform(
            e, viet_texts, device
        )  

        batch_size, seq_len = x.shape
        if y_shift is not None: # Training mode
            if mask is None:
                raise ValueError("Mask is required for training mode.")
            # Embeddings
            embed_y = self.embedding(
                y_shift
            )  # embed_y shape: (batch_size, seq_len, embed_dim)
            dec_h = torch.zeros(batch_size, self.hidden_size, device=device)
            dec_c = torch.zeros(batch_size, self.hidden_size, device=device)
            dec_out_list = []
            
            # Feeding through RNNs
            for t in range(seq_len):
                input_t = embed_y[:, t, :]
                dec_h, dec_c = self.decoder_lstm(input_t, (dec_h, dec_c))
                dec_out_list.append(dec_h)
            dec_out = torch.stack(
                dec_out_list, dim=1
            )  # dec_out shape: (batch_size, seq_len, hidden_size)
            
            # Bahdanau-style attention
            u = self.dropout_u(self.U(dec_out))  # u shape: (batch_size, seq_len, att_dim)
            tilde_h_exp = tilde_h.unsqueeze(
                1
            )  # tilde_h_exp shape: (batch_size, 1, seq_len, att_dim)
            u_exp = u.unsqueeze(2)  # u_exp shape: (batch_size, seq_len, 1, att_dim)
            tanh_in = torch.tanh(
                tilde_h_exp + u_exp
            )  # tanh_in shape: (batch_size, seq_len, seq_len, att_dim)
            a = self.v(tanh_in).squeeze(3)  # a shape: (batch_size, seq_len, seq_len)
            a = self.dropout_v(a)
            mask_exp = mask.unsqueeze(1)  # mask_exp shape: (batch_size, 1, seq_len)
            a = a.masked_fill(
                mask_exp == 0, -1e9
            )  # a shape: (batch_size, seq_len, seq_len)
            alpha = F.softmax(a, dim=2)  # alpha shape: (batch_size, seq_len, seq_len)
            c = torch.matmul(
                alpha, raw_enc
            )  # c shape: (batch_size, seq_len, self.transform.hidden_dim)
            
            # Inference
            out_cat = torch.cat(
                (dec_out, c), dim=2
            )  # out_cat shape: (batch_size, seq_len, hidden_size + self.transform.hidden_dim)
            logits = self.output_linear(
                out_cat
            )  # logits shape: (batch_size, seq_len, vocab_size)
            
            return logits, l
        else: # Inference mode 
            if mask is None:
                mask = (x != PAD_ID).float()  # mask shape: (batch_size, seq_len)

            dec_h = torch.zeros(batch_size, self.hidden_size, device=device)
            dec_c = torch.zeros(batch_size, self.hidden_size, device=device)
            current_y = torch.full(
                (batch_size, 1), SOS_ID, dtype=torch.long, device=device
            )
            logits_list = []
            
            for _ in range(seq_len):
                embed = self.embedding(current_y).squeeze(
                    1
                )  # embed shape: (batch_size, embed_dim)
                dec_h, dec_c = self.decoder_lstm(embed, (dec_h, dec_c))
                
                # Bahdanau-style attention
                u = self.U(dec_h)  # (batch_size, att_dim)
                tanh_in = torch.tanh(tilde_h + u.unsqueeze(1))        # (batch_size, seq_len, att_dim)
                a = self.v(tanh_in).squeeze(2)                        # (batch_size, seq_len)
                a = a.masked_fill(mask == 0, -1e9)
                alpha = F.softmax(a, dim=1)                           # (batch_size, seq_len)
                
                # Weighted sum of encoder states
                alpha = alpha.unsqueeze(1)                            # (batch_size, 1, seq_len)
                c = torch.bmm(alpha, raw_enc).squeeze(1)              # (batch_size, hidden_dim)
                
                out_cat = torch.cat(
                    (dec_h, c), dim=1
                )  # out_cat shape: (batch_size, hidden_size + self.transform.hidden_size)
                logit = self.output_linear(
                    out_cat
                )  # logit shape: (batch_size, vocab_size)
                logits_list.append(logit)
                pred = torch.argmax(logit, dim=1)  # pred shape: (batch_size,)
                current_y = pred.unsqueeze(1)  # current_y shape: (batch_size, 1)
            logits = torch.stack(
                logits_list, dim=1
            )  # logits shape: (batch_size, seq_len, vocab_size)
            return logits, l