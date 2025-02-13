import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.TDformer_EncDec import Encoder, EncoderLayer,iEncoder, AttentionLayer
from layers.Attention import FullAttention
from layers.RevIN import RevIN
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.iencoder = iEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 1724, 4),
                    1724,
                    1724,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(1724)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector1 = nn.Linear(862, 1724, bias=True)
        self.projector2 = nn.Linear(1724, 862, bias=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.revin_trend = RevIN(96).to(self.device)
        self.channel_attention = nn.Sequential(
            nn.Linear(862, int(862 / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(862 / 4), 862)
        )
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        trend_enc = x_enc.permute(0, 2, 1)
        trend_enc = self.revin_trend(trend_enc, 'norm')  #(16,10,42)
        trend_enc = trend_enc.permute(0, 2, 1)
        
        x_enc = trend_enc

        _, _, N = x_enc.shape # B L N
        #print('x_enc.shape:',x_enc.shape)#(16,96,862)
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        #print('enc_out1.shape',enc_out.shape)#(16,866,512),加上mark后
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        #print(enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        #print('enc_out2.shape',enc_out.shape)#(16,866,512)
        

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        # (16,96,862)
        

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #print('dec_out2:',dec_out.shape)#(16,96,862)
        #print('dec_out[:, -self.pred_len:, :]:',dec_out[:, -self.pred_len:, :].shape)#(16,96,862)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]