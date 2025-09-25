from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn


class PhoneticEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError("Subclasses must implement the hidden_size property.")


class PhoBERTEncoder(PhoneticEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.viet_encoder = AutoModel.from_pretrained("vinai/phovert-base-v2", torchscript=True)
        self.viet_encoder.eval()
        for param in self.viet_encoder.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phovert-base-v2")

    @property
    def hidden_size(self):
        return self.viet_encoder.config.hidden_size

    def forward(self, viet_texts, seq_len, device):
        batch_size = len(viet_texts)

        inputs = self.tokenizer(
            viet_texts, padding=True, return_tensors="pt", return_offsets_mapping=True
        )
        out = self.viet_encoder(**inputs)

        hidden = (
            out.last_hidden_state
        )  # shape: (batch_size, max_subword_len, viet_hidden_size

        padded = torch.zeros(
            batch_size, seq_len, self.viet_encoder.config.hidden_size, device=device
        )

        for i, length in enumerate(inputs["attention_mask"].sum(dim=1)):
            padded[i, :length, :] = hidden[i, :length, :]

        return padded


class BARTphoEncoder(PhoneticEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.viet_encoder = AutoModel.from_pretrained("vinai/bartpho-syllable", torchscript=True)
        self.viet_encoder.eval()
        for param in self.viet_encoder.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    @property
    def hidden_size(self):
        return self.viet_encoder.config.hidden_size

    def forward(self, viet_texts, seq_len, device):
        batch_size = len(viet_texts)

        inputs = self.tokenizer(
            viet_texts, padding=True, return_tensors="pt", return_offsets_mapping=True
        )
        out = self.viet_encoder(**inputs)

        hidden = (
            out.last_hidden_state
        )  # shape: (batch_size, max_subword_len, viet_hidden_size

        padded = torch.zeros(
            batch_size, seq_len, self.viet_encoder.config.hidden_size, device=device
        )

        for i, length in enumerate(inputs["attention_mask"].sum(dim=1)):
            padded[i, :length, :] = hidden[i, :length, :]

        return padded
