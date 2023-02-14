import torch.nn as nn
import encoder_decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, d_k, d_v, d_ff, en_n_layers=2, de_n_layers=2, n_heads=1):
        """
        src_vocab_size: 输入字典大小 在翻译项目中，可认为是被翻译的文字词典大小
        tgt_vocab_size: 输出字典大小 在翻译项目中，可认为是翻译目标文字词典大小
        d_model: 词向量维度
        d_k: Q和K的维度
        d_v: V的维度
        n_heads: 多头注意力的头数
        d_ff: 前馈神经网络的中转层数
        en_n_layers: 编码器的层数
        de_n_layers: 解码器的层数
        """
        
        super(Transformer, self).__init__()
        self.encode = encoder_decoder.Encoder(src_vocab_size, d_model, d_k, d_v, n_heads, d_ff, en_n_layers)
        self.decode = encoder_decoder.Decoder(tgt_vocab_size, d_model, d_k, d_v, n_heads, d_ff, de_n_layers)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self,  enc_inputs, dec_inputs):

        enc_outputs, enc_self_attns = self.encode(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns =  self.decode(dec_inputs, dec_inputs, enc_outputs) 
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns