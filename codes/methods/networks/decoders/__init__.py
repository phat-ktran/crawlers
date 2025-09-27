import torch
from torch import nn

from codes.methods.networks.encoders.fuse import Fusion
from codes.methods.utils import SOS_ID


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        context_dim: int,
        vocab_size: int,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_lstm = nn.LSTMCell(embed_dim, hidden_size)
        self.output_linear = nn.Linear(hidden_size + context_dim, vocab_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)  # input -> hidden
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)  # hidden -> hidden
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)

    def attend(
        self,
        dec_out,
        attention,
        src_enc,
        fused_enc,
        src_mask,
        ref_mask,
    ):
        return attention(
            queries=dec_out, keys=fused_enc, values=fused_enc, mask=src_mask
        )  # (batch_size, seq_len, context_dim)

    def forward(
        self,
        y_shift,
        embedding,
        attention,
        src_enc,
        fused_enc,
        src_mask,
        ref_mask,
        device,
        max_len=None,
    ):
        """
        This is the forward method for the MultiSourceDecoder class, which extends the Decoder class to handle multi-source inputs.

        Args:
            y_shift (torch.Tensor): The shifted target sequence used for teacher forcing during training. Shape: (batch_size, seq_len).
            embedding (nn.Module): The embedding layer that converts token indices into dense vectors.
            attention (nn.Module): The attention mechanism used to compute context vectors.
            src_enc (torch.Tensor): The source encoder outputs. Shape: (batch_size, src_seq_len, hidden_size).
            fused_enc (torch.Tensor): The fused encoder outputs. Shape: (batch_size, fused_seq_len, hidden_size).
            src_mask (torch.Tensor): The mask for the source sequence to ignore padding tokens. Shape: (batch_size, src_seq_len).
            ref_mask (torch.Tensor): The mask for the reference sequence to ignore padding tokens. Shape: (batch_size, ref_seq_len).
            device (torch.device): The device on which tensors are allocated.
            max_len (int, optional): The maximum length of the output sequence during inference. Required in inference mode.

        Returns:
            torch.Tensor: The output logits for each token in the vocabulary.
                          Shape: (batch_size, seq_len, vocab_size) in training mode,
                                 (batch_size, max_len, vocab_size) in inference mode.
        """
        batch_size = y_shift.size(0) if y_shift is not None else fused_enc.size(0)
        dec_h = torch.zeros(batch_size, self.hidden_size, device=device)
        dec_c = torch.zeros(batch_size, self.hidden_size, device=device)

        if y_shift is not None:
            # Training mode
            embed_y = embedding(y_shift)  # (batch_size, seq_len, embed_dim)
            seq_len = embed_y.size(1)
            dec_out_list = []
            for t in range(seq_len):
                input_t = embed_y[:, t, :]
                dec_h, dec_c = self.decoder_lstm(input_t, (dec_h, dec_c))
                dec_out_list.append(dec_h)

            dec_out = torch.stack(
                dec_out_list, dim=1
            )  # (batch_size, seq_len, hidden_size)

            context = self.attend(
                dec_out, attention, src_enc, fused_enc, src_mask, ref_mask
            )

            out_cat = torch.cat((dec_out, context), dim=2)
            logits = self.output_linear(out_cat)
            return logits
        else:
            # Inference mode
            if max_len is None:
                raise ValueError("max_len is required for inference mode.")
            current_y = torch.full(
                (batch_size,), SOS_ID, dtype=torch.long, device=device
            )
            logits_list = []
            for _ in range(max_len):
                embed = embedding(current_y.unsqueeze(1)).squeeze(
                    1
                )  # (batch_size, embed_dim)
                dec_h, dec_c = self.decoder_lstm(embed, (dec_h, dec_c))
                context = self.attend(
                    dec_h, attention, src_enc, fused_enc, src_mask, ref_mask
                )
                out_cat = torch.cat((dec_h, context), dim=1)
                logit = self.output_linear(out_cat)  # (batch_size, vocab_size)
                logits_list.append(logit)
                current_y = torch.argmax(logit, dim=1)
            logits = torch.stack(
                logits_list, dim=1
            )  # (batch_size, max_len, vocab_size)
            return logits


class MultiSourceDecoder(Decoder):
    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        context_dim: int,
        vocab_size: int,
        fuser: Fusion,
        **kwargs,
    ):
        super().__init__(embed_dim, hidden_size, context_dim, vocab_size, **kwargs)
        self.fuser = fuser
        self.output_linear = nn.Linear(hidden_size + fuser.fused_dim, vocab_size)

    def attend(self, dec_out, attention, src_enc, fused_enc, src_mask, ref_mask):
        src_ctx = attention(
            queries=dec_out, keys=src_enc, values=src_enc, mask=src_mask
        )  # (batch_size, seq_len, 2*hidden_size)

        ref_ctx = attention(
            queries=dec_out, keys=fused_enc, values=fused_enc, mask=ref_mask
        )  # (batch_size, seq_len, context_dim)

        return self.fuser(src_ctx, ref_ctx, src_mask, src_mask)
