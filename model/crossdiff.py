import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList
import copy
from model.mdm import PositionalEncoding, TimestepEmbedder

class CrossDiff(nn.Module):
    def __init__(self, args):
        super().__init__()

        
        if args.dataset == 'kit':
            self.njoints_m = 251
            self.njoints_j = 128
        else:
            self.njoints_m = 263
            self.njoints_j = 134
        


        self.latent_dim = args.latent_dim # 512

        self.ff_size = 1024
        self.num_layers = args.layers
        self.num_layers2 = args.layers2
        self.num_heads = 4
        self.dropout = 0.1

        self.activation = 'gelu'
        self.clip_dim = 512
        self.action_emb = 'tensor'

        self.input_feats_m = self.njoints_m
        self.input_feats_j = self.njoints_j


        self.cond_mode = 'text'
        self.cond_mask_prob = args.cond_mask_prob

        # pose pipeline
        self.p_linear_m = nn.Linear(self.input_feats_m-4, self.latent_dim)
        self.p_linear_j = nn.Linear(self.input_feats_j-2, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder_m = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        self.seqTransEncoder_j = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        
        self.motion_token = nn.Parameter(torch.randn((196, 1, self.latent_dim)))
        self.joint_token = nn.Parameter(torch.randn((196, 1, self.latent_dim)))
        self.decoder_m = ModuleList([copy.deepcopy(seqTransDecoderLayer) for i in range(self.num_layers2)])
        self.decoder_j = ModuleList([copy.deepcopy(seqTransDecoderLayer) for i in range(self.num_layers2)])
        self.bridge = ModuleList([copy.deepcopy(seqTransEncoderLayer) for i in range(self.num_layers2)])

        # root pipeline
        r_EncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=512,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        r_DecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=512,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.r_linear_m = nn.Linear(4, self.latent_dim)
        self.r_encoder_m = nn.TransformerEncoder(r_EncoderLayer,
                                                        num_layers=2)
        self.r_decoder_m = ModuleList([copy.deepcopy(r_DecoderLayer) for i in range(2)])
        self.r_linear2_m = nn.Linear(self.latent_dim, 4)
        self.r_linear_j = nn.Linear(2, self.latent_dim)
        self.r_encoder_j = nn.TransformerEncoder(r_EncoderLayer,
                                                        num_layers=2)
        self.r_decoder_j = ModuleList([copy.deepcopy(r_DecoderLayer) for i in range(2)])
        self.r_linear2_j = nn.Linear(self.latent_dim, 2)


        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

        self.p_linear2_m = nn.Linear(self.latent_dim, self.input_feats_m-4)
        self.p_linear2_j = nn.Linear(self.latent_dim, self.input_feats_j-2)




    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, x, timesteps, y, force_mask=False, return_m=True, return_j=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        x = x[:,:,0].permute(2,0,1)

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        enc_text = y
        emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        if njoints == 263 or njoints == 251:
            x_root = self.r_linear_m(x[:,:,:4])
            x_pose = self.p_linear_m(x[:,:,4:])
        else:
            x_root = self.r_linear_j(x[:,:,:2])
            x_pose = self.p_linear_j(x[:,:,2:])

        xseq_pose = torch.cat((emb, x_pose), axis=0)  # [seqlen+1, bs, d]
        xseq_pose = self.sequence_pos_encoder(xseq_pose)  # [seqlen+1, bs, d]
        xseq_root = torch.cat((emb, x_root), axis=0)  
        xseq_root = self.sequence_pos_encoder(xseq_root)

        if njoints == 263 or njoints == 251:
            xseq_pose = self.seqTransEncoder_m(xseq_pose)   # , src_key_padding_mask=~maskseq)  # [seqlen+1, bs, d]
            xseq_root = self.r_encoder_m(xseq_root)
        else:
            xseq_pose = self.seqTransEncoder_j(xseq_pose)
            xseq_root = self.r_encoder_j(xseq_root)

        middle_infos = []
        for mod in self.bridge:
            xseq_pose = mod(xseq_pose)
            middle_infos.append(xseq_pose)
            
        output_m = None
        output_j = None
        if return_m:
            output_m = self.motion_token.expand((-1, bs, -1))
            r_output_m = xseq_root[1:]
            for i, (info, mod) in enumerate(zip(middle_infos, self.decoder_m)):
                output_m = mod(output_m, info)
                if i >= self.num_layers2 - 2:
                    r_output_m = self.r_decoder_m[i + 2 - self.num_layers2](r_output_m, output_m)
            output_m = self.p_linear2_m(output_m) # [nframes, bs, nfeats]
            r_output_m = self.r_linear2_m(r_output_m)
            output_m = torch.cat([r_output_m, output_m], dim=-1) 
            output_m = output_m.permute(1,2,0).unsqueeze(2)
        
        if return_j:
            output_j = self.joint_token.expand((-1, bs, -1))
            r_output_j = xseq_root[1:]
            for i, (info, mod) in enumerate(zip(middle_infos, self.decoder_j)):
                output_j = mod(output_j, info)
                if i >= self.num_layers2 - 2:
                    r_output_j = self.r_decoder_j[i + 2 - self.num_layers2](r_output_j, output_j)
            output_j = self.p_linear2_j(output_j) # [nframes, bs, nfeats]
            r_output_j = self.r_linear2_j(r_output_j)
            output_j = torch.cat([r_output_j, output_j], dim=-1) 
            output_j = output_j.permute(1,2,0).unsqueeze(2)

        return {'m': output_m, 'j': output_j}
    
    