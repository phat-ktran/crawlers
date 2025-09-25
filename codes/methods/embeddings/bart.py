import torch
from transformers import AutoModel, AutoTokenizer

#BARTpho-syllable
syllable_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
bartpho_syllable = AutoModel.from_pretrained("vinai/bartpho-syllable")
TXT = ''  
input_ids = syllable_tokenizer(TXT, return_tensors='pt')['input_ids']

print(input_ids)

with torch.no_grad():
    features = bartpho_syllable(input_ids)
    print(features.last_hidden_state.shape)