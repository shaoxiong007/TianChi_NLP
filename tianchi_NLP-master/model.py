import torch.nn as nn
import torch
import encoder_decoder
import torch.nn.functional as F

class mymodel(nn.Module):
    def __init__(self, vocab, d_model, d_k, d_v, n_heads, d_ff, n_layers):
    
        super(mymodel, self).__init__()
        self.Mencode = encoder_decoder.Encoder(len(vocab.voc), d_model, d_k, d_v, n_heads, d_ff, n_layers)
        self.gru = nn.GRU( d_model, 50, 3, dropout = 0.2,
                          batch_first = True, bidirectional = True)
        self.tanh = nn.Tanh()
        self.l1 = nn.Linear(120,80)
        self.l2 = nn.Linear(80,14)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(100,14)
        
    def forward(self, input):
        enc_outputs, enc_self_attns = self.Mencode(input)
        out, state = self.gru(enc_outputs)
        output_fw = state[-2,:,:]
        output_bw = state[-1,:,:]
        output1 = torch.cat([output_fw,output_bw],dim = -1)
        out = self.l3(self.relu(output1))
        return F.log_softmax(out,dim = -1)