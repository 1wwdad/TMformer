from layers.TDformer_EncDec import EncoderLayer, Encoder,iEncoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding , DataEmbedding_inverted
from layers.Attention import  FourierAttention, FullAttention  #WaveletAttention,
from layers.RevIN import RevIN
import torch.nn.functional as F



class Model(nn.Module):
    """
    Transformer for seasonality, MLP for trend
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decomp
        kernel_size = configs.moving_avg    #24
        if isinstance(kernel_size, list):
            #yes
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_seasonal_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # iEmbedding
        self.enc_embedding = DataEmbedding_inverted(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder
        
        self.encoder = iEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        #FourierAttention(T=configs.temp, activation=configs.activation,
                                                  #output_attention=configs.output_attention),configs.d_model, configs.n_heads),
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
        
        if configs.version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len,
                                                  seq_len_kv=configs.seq_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                  seq_len_kv=configs.seq_len // 2 + configs.pred_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
            
        elif configs.version == 'Fourier':
            enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Time':
            enc_self_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_self_attention = FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_cross_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        enc_self_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        dec_self_attention,
                        configs.d_model),
                    AttentionLayer(
                        dec_cross_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # Encoder
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

        self.revin_trend = RevIN(192).to(self.device)
        
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector1 = nn.Linear(2, 1, bias=True)

        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size

        self.conv_layer =  nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3,3), padding=1),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3,3), padding=1),
        )

        self.conv_layer1 =  nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3,3), padding=1),
        )
        '''
        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(9, 9), padding=4),
        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(11, 11), padding=5),
        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(13, 13), padding=6),
        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(15, 15), padding=7),
        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(17, 17), padding=8),
        '''
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(5, 5), padding=2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(7, 7), padding=3),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(9, 9), padding=4),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(11, 11), padding=5),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(13, 13), padding=6),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(15, 15), padding=7),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(17, 17), padding=8),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(19, 19), padding=9),
        ])
        self.conv_combine = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=2, kernel_size=1),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)  # cuda()

        seasonal_enc, trend_enc = self.decomp(x_enc)  


        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # seasonal
        enc_out = self.enc_seasonal_embedding(seasonal_enc, x_mark_enc)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask)


        dec_out = self.dec_seasonal_embedding(seasonal_dec, x_mark_dec)
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        seasonal_out = seasonal_out[:, -self.pred_len:, :]


        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        '''
        trend_enc = trend_enc.permute(0, 2, 1)
        trend_enc = self.revin_trend(trend_enc, 'norm')  #(16,10,42)
        trend_enc = trend_enc.permute(0, 2, 1)
        '''
        #trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1)

        _, _, N = trend_enc.shape # B L N
        trend_out = self.enc_embedding(trend_enc, x_mark_enc) #([16, 43, 512])
        #print(trend_out.shape)

        #print('trend_enc',trend_enc.shape)            #([32, 20, 1])
        trend_out, attn = self.encoder(trend_out, attn_mask=None)  #([16, 25, 512])

        trend_out =  self.projector(trend_out).permute(0, 2, 1)[:, :, :N] # filter the covariates ([16, 96, 862])

        ###
        '''
        _,M,_ = trend_out.shape
        #print(trend_out.shape)
        trend_out = self.kan(trend_out.view(-1,512))  #(40,21)
        #print(trend_out.shape)
        trend_out = trend_out.view(-1,M,96).permute(0, 2, 1)[:, :, :N]
        ###
        '''
        '''
        trend_out = trend_out.permute(0, 2, 1)
        trend_out = self.revin_trend(trend_out, 'denorm')
        trend_out = trend_out.permute(0, 2, 1)
        '''
        # final
        #dec_out = trend_out + seasonal_ratio * seasonal_out
        #dec_out = trend_out 

        
        trend_out1=trend_out
        seasonal_out1=seasonal_out
        
        
        trend_out = trend_out.unsqueeze(1)   #([16, 1, 96, 862])
        seasonal_out = seasonal_out.unsqueeze(1)
        concatenated = torch.cat([seasonal_out, trend_out], dim=1) #([16, 2, 96, 862])
        conv_out = self.conv_layer(concatenated)

        _,_,_,M = concatenated.shape
        dec_out = conv_out.view(self.batch_size, self.pred_len,M, -1)
        dec_out = self.projector1(dec_out).view(self.batch_size,self.pred_len,M)
        dec_out = dec_out[:,:,-1:]
        
        '''
        trend_out = trend_out.unsqueeze(1)
        seasonal_out = seasonal_out.unsqueeze(1)
        concatenated = torch.cat([seasonal_out, trend_out], dim=1)  
        
        multi_scale_outputs = []
        for conv in self.multi_scale_conv:
            conv_out = conv(concatenated)  
            multi_scale_outputs.append(conv_out)
        
        conv_out = torch.cat(multi_scale_outputs, dim=1)  
        
        conv_out = self.conv_combine(conv_out)  
        

        _, _, L, D = conv_out.shape
        dec_out = conv_out.view(self.batch_size, self.pred_len, D, -1)
        dec_out = self.projector1(dec_out).view(self.batch_size, self.pred_len, D)
        dec_out = dec_out[:, :, -1:]
        '''

        
        if self.output_attention:
            return dec_out, trend_enc, seasonal_enc, trend_out1, seasonal_out1, attn_e, attn_d,attn
        elif self.output_stl:
            return dec_out, trend_enc, seasonal_enc, trend_out1, seasonal_out1, attn_e, attn_d,attn
        else:
            return dec_out
