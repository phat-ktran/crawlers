import torch
from transformers import AutoModel, AutoTokenizer
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir='/tmp/')

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/tmp/')

text = "phụ mạc đăng doanh bát niên phúc hải lục niên phúc nguyên nhị niên trang tông dụ hoàng đế huý ninh hựu huý huyến tại vị thập lục niên thọ tam thập tứ tuế đế tao đốn tị nạn lại cựu thần tôn lập ngoại kết lân phong nội nhiệm hiền tướng cố nhân giai lạc dụng trung hưng chi cơ thực triệu ư thử hĩ đế nãi chiêu tông chi tử thánh tông chi huyền tôn"

sentence = rdrsegmenter.word_segment(text)[0]

print(sentence)

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

print(tokenizer.encode(sentence))

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples
    
    print(features.last_hidden_state.shape)