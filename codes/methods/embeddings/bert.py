import torch
from transformers import AutoModel, AutoTokenizer
import py_vncorenlp

# Download vncorenlp if not already present
py_vncorenlp.download_model(save_dir="/tmp/")
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="/tmp/")

texts = [
    "phụ mạc đăng doanh bát niên phúc hải lục niên phúc nguyên nhị niên",
    "trang tông dụ hoàng đế tuế đế tao hiền tướng cố nhân giai lạc dụng",
    "trung hưng chi cơ thực triệu ư thử hĩ đế nãi chiêu tông chi tử thánh tông chi huyền tôn",
    "huý ninh hựu huý huyến tại vị thập lục niên thọ tam thập tứ",
    "đốn tị nạn lại cựu thần tôn lập ngoại kết lân phong nội nhiệm",
]

# Segment and re-join into space-separated strings
sentences = [" ".join(rdrsegmenter.word_segment(text)) for text in texts]

# Load PhoBERT
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Batch tokenize
inputs = tokenizer(
    sentences,
    padding=True,          # pad to longest sequence in batch
    truncation=True,       # cut off if longer than max_length
    return_tensors="pt"    # return PyTorch tensors
)

print(inputs["input_ids"].shape)  # (batch_size, max_seq_len)
print(inputs["input_ids"])

# Forward pass (batched)
with torch.no_grad():
    outputs = phobert(**inputs)

print(outputs.last_hidden_state.shape)  # (batch_size, max_seq_len, hidden_size)

