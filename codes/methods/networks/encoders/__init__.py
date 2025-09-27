from torch import nn
from codes.methods.networks.encoders.fuse import Fusion
from codes.methods.networks.encoders.sinonom_enc import SinoNomEncoder
from codes.methods.networks.encoders.bert_enc import PhoBERTEncoder, VietEncoder

class Encoder(nn.Module):
    def __init__(self, src_enc: SinoNomEncoder, ref_enc: VietEncoder, fuser: Fusion):
        super().__init__()
        self.src_encoder = src_enc
        self.ref_encoder = ref_enc
        self.fuser = fuser
        
    def init_weights(self):
        self.src_encoder.init_weights()
        self.ref_encoder.init_weights()

    def forward(self, embed_x, viet_texts, src_mask):
        src_enc, _ = self.src_encoder(embed_x)
        ref_enc, ref_mask = self.ref_encoder(viet_texts, src_enc.device)
        
        if ref_enc is None:
            return src_enc, src_enc
        return src_enc, self.fuser(src_enc, ref_enc, src_mask, ref_mask)
    