import os
import torch
from torch import nn
from torch.utils.data import Dataset

from codes.methods.networks.encoders import Encoder
from codes.methods.networks.decoders import Decoder
from codes.methods.networks.attentions import BahdanauAttention, LightWeightQT, ComplexQT
from codes.methods.embeddings.cbow import CBOW

from codes.methods.utils import SOS_ID, UNK_ID, PAD_ID, EOS_ID


class SinoNomDataset(Dataset):
    def __init__(self, lines, vocab, bypass_check=False):
        self.data = []

        for line in lines:
            parts = line.strip().split("\t")
            input, target = parts[0], parts[1]
            x = [vocab.get(c, UNK_ID) for c in input.strip()]
            y = [vocab.get(c, UNK_ID) for c in target.strip()]
            b = [1 if ci == ct else 0 for ci, ct in zip(input, target)]
            # 🔑 Append EOS to target
            y.append(EOS_ID)

            b_prime = [int(bb) for bb in parts[2].split(",")]
            t_text = parts[3]

            if not bypass_check:
                assert len(x) == len(y) - 1 == len(b_prime) == len(t_text.split()), (
                    f"Data mismatch: x({parts[0]}), y({parts[1]}), b({parts[2]}), t_text({parts[3]})"
                )

            self.data.append((x, y, b_prime, b, t_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    xs, ys, bps, bs, ts = zip(*batch)
    lens = [len(y) for y in ys]  # use y length (includes EOS)
    max_l = max(lens)

    pad_x = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    pad_y = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    pad_y_shift = torch.full((len(batch), max_l), PAD_ID, dtype=torch.long)
    pad_b = torch.zeros((len(batch), max_l), dtype=torch.float)
    pad_bp = torch.zeros((len(batch), max_l), dtype=torch.float)
    mask = torch.zeros((len(batch), max_l), dtype=torch.float)

    for i in range(len(batch)):
        l = lens[i]
        pad_x[i, : len(xs[i])] = torch.tensor(xs[i])
        pad_y[i, :l] = torch.tensor(ys[i])
        pad_b[i, : len(bs[i])] = torch.tensor(bs[i], dtype=torch.float)
        pad_bp[i, : len(bps[i])] = torch.tensor(bps[i], dtype=torch.float)

        mask[i, :l] = 1  # EOS is included in mask

        # Teacher forcing input: SOS + shifted y (without last EOS)
        pad_y_shift[i, 0] = SOS_ID
        pad_y_shift[i, 1:l] = torch.tensor(ys[i][:-1])

    return pad_x, pad_y_shift, pad_y, pad_bp, pad_b, list(ts), mask


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder: Encoder,
        decoder: Decoder,
        attention: BahdanauAttention,
        drop_rate=0.5,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = encoder.src_encoder.embed_dim
        self.embedding = nn.Embedding(
            vocab_size, encoder.src_encoder.embed_dim, padding_idx=PAD_ID
        )
        self.dropout_e = nn.Dropout(p=drop_rate)
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def init_weights(self) -> None:
        self.encoder.init_weights()
        self.decoder.init_weights()

    def load_pretrained_embeddings(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained embeddings file not found at {path}")
        pretrained_weights = torch.load(path, map_location=torch.device("cpu"))
        cbow = CBOW(self.vocab_size - 4, self.emb_dim)
        cbow.load_state_dict(pretrained_weights)
        self.embedding.weight.data[-(self.vocab_size - 4) :, :].copy_(
            cbow.in_embed.weight.data[:-1, :]
        )

    def forward(self, x, viet_texts, pad_b, y_shift=None, src_mask=None):
        device = x.device
        _, seq_len = x.shape

        embed_x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embed_x = self.dropout_e(embed_x)

        if src_mask is None and y_shift is None:
            src_mask = (x != PAD_ID).float()  # (batch_size, seq_len)

        src_enc, fused_enc, ref_mask = self.encoder(embed_x, viet_texts, src_mask)

        logits = self.decoder(
            y_shift,
            self.embedding,
            self.attention,
            src_enc,
            fused_enc,
            src_mask,
            ref_mask,
            device,
            max_len=seq_len if y_shift is None else None,
        )

        return logits, None


class QTSeq2Seq(Seq2Seq):
    def __init__(
        self,
        vocab_size: int,
        encoder: Encoder,
        decoder: Decoder,
        attention: BahdanauAttention,
        drop_rate=0.5,
        max_len=40,
        qt_dim=160,
        qt_heads=8,
        qt_mod="LightWeightQT",
        **kwargs,
    ):
        super().__init__(vocab_size, encoder, decoder, attention, drop_rate, **kwargs)
        assert qt_mod in ["LightWeightQT", "ComplexQT"]
        assert self.encoder.fuser is not None
        self.qts = eval(qt_mod)(
            d_model=qt_dim,
            src_hidden_size=self.encoder.fuser.fused_dim,
            max_len=max_len,
            qt_heads=qt_heads,
        )

    def forward(self, x, viet_texts, pad_b, y_shift=None, src_mask=None):
        device = x.device
        _, seq_len = x.shape

        embed_x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embed_x = self.dropout_e(embed_x)

        if src_mask is None and y_shift is None:
            src_mask = (x != PAD_ID).float()  # (batch_size, seq_len)

        src_enc, fused_enc, ref_mask = self.encoder(embed_x, viet_texts, src_mask)

        l = self.qts(fused_enc, src_mask, pad_b)

        logits = self.decoder(
            y_shift,
            self.embedding,
            self.attention,
            src_enc,
            fused_enc,
            src_mask,
            ref_mask,
            device,
            max_len=seq_len if y_shift is None else None,
        )

        return logits, l
