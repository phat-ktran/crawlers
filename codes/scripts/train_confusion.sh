uv run -m codes.methods.train_confusion \
    --train data/postprocess/train.txt \
    --val data/postprocess/val.txt \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --confusion data/postprocess/similarity.csv \
    --output-dir data/postprocess/models/confusion_net/ \
    --vocab data/postprocess/vocab.txt \
    --pretrained-embs data/postprocess/embeddings/best.pt