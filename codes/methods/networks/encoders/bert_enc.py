from torch import nn
import torch
from transformers import AutoTokenizer, AutoModel


class VietEncoder(nn.Module):
    def init_weights(self):
        pass
        
    def forward(self, viet_texts, device):
        raise NotImplementedError("This method should be implemented by subclasses.")
        
class Identity(VietEncoder):
    def forward(self, viet_texts, device):
        return None, None

class PhoBERTEncoder(VietEncoder):
    def __init__(self, model_name="vinai/phobert-base-v2"):
        super().__init__()
        self.viet_encoder = AutoModel.from_pretrained(model_name)
        self.viet_encoder.eval()
        for param in self.viet_encoder.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def init_weights(self):
        pass

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
