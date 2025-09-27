import torch
from torch import nn

from codes.methods.utils import SOS_ID


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_size: int,
        context_dim: int,
        vocab_size: int,
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

    def forward(
        self, y_shift, embedding, attention, src_enc, fused_enc, src_mask, device, max_len=None
    ):
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
            
            context = attention(
                queries=dec_out,
                keys=fused_enc,
                values=fused_enc,
                mask=src_mask
            )  # (batch_size, seq_len, context_dim)
            
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
                context = attention(
                   queries=dec_h, keys=fused_enc, values=fused_enc, mask=src_mask
                )  # (batch_size, context_dim)
                out_cat = torch.cat((dec_h, context), dim=1)
                logit = self.output_linear(out_cat)  # (batch_size, vocab_size)
                logits_list.append(logit)
                current_y = torch.argmax(logit, dim=1)
            logits = torch.stack(
                logits_list, dim=1
            )  # (batch_size, max_len, vocab_size)
            return logits
