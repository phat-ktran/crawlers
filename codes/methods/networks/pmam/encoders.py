from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn


class PhoneticEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError("Subclasses must implement the hidden_size property.")


class PhoBERTEncoder(nn.Module):
    def __init__(self, model_name="vinai/phobert-base-v2"):
        super().__init__()
        self.viet_encoder = AutoModel.from_pretrained(model_name)
        self.viet_encoder.eval()
        for param in self.viet_encoder.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def hidden_size(self):
        return self.viet_encoder.config.hidden_size

    def forward(self, viet_texts, device):
        with torch.no_grad():
            inputs = self.tokenizer(
                viet_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            out = self.viet_encoder(**inputs)
            hidden = out.last_hidden_state  # (batch, subword_len, hidden_size)
        return hidden, inputs["attention_mask"]


class BARTphoEncoder(PhoneticEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.viet_encoder = AutoModel.from_pretrained(
            "vinai/bartpho-syllable", torchscript=True
        )
        self.viet_encoder.eval()
        for param in self.viet_encoder.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    @property
    def hidden_size(self):
        return self.viet_encoder.config.hidden_size

    def forward(self, viet_texts, seq_len, device):
        batch_size = len(viet_texts)
        with torch.no_grad():
            inputs = self.tokenizer(
                viet_texts,
                padding=True,  # pad to longest sequence in batch
                truncation=True,  # cut off if longer than max_length
                return_tensors="pt",  # return PyTorch tensors
            )
            attn_mask = inputs["attention_mask"]
            out = self.viet_encoder(**inputs)
            hidden = (
                out.last_hidden_state
            )  # shape: (batch_size, max_subword_len, viet_hidden_size

            padded_hidden = torch.zeros(
                batch_size, seq_len, self.hidden_size, device=hidden.device
            )

            # Copy each sequence up to its real length
            lengths = attn_mask.sum(dim=1)  # number of valid tokens per sentence
            for i, length in enumerate(lengths):
                padded_hidden[i, :length, :] = hidden[i, :length, :]

        return padded_hidden
