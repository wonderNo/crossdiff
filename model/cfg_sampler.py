import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        # assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'
        # if args.mode in ['motion','motion_uselift']:

        #     # pointers to inner model
        #     # self.rot2xyz = self.model.rot2xyz
        #     self.translation = self.model.translation
        #     self.njoints = self.model.njoints
        #     self.nfeats = self.model.nfeats
        #     self.data_rep = self.model.data_rep
        #     self.cond_mode = self.model.cond_mode
        # else:
            # self.rot2xyz = self.model.mdm3d.rot2xyz
            # self.translation = self.model.mdm3d.translation
            # self.njoints = self.model.mdm3d.njoints
            # self.nfeats = self.model.mdm3d.nfeats
            # self.data_rep = self.model.mdm3d.data_rep
            # self.cond_mode = self.model.mdm3d.cond_mode

    def forward(self, x, timesteps, y=None,
                                       return_m=True, return_j=False):
        # cond_mode = self.model.cond_mode
        # assert cond_mode in ['text', 'action']
        # y_uncond = deepcopy(y)
        # y_uncond['uncond'] = True
        if 'force_mask' in y.keys():
            out = self.model(x, timesteps, y['enc_text'], return_m=return_m,
                            return_j=return_j, force_mask=True)
        else:
            out = self.model(x, timesteps, y['enc_text'], return_m=return_m,
                            return_j=return_j)
            if 'scale' in y.keys():
                out_uncond = self.model(x, timesteps, y['enc_text'], return_m=return_m,
                            return_j=return_j, force_mask=True)
                if return_m:
                    out['m'] = out_uncond['m'] + (y['scale'].view(-1, 1, 1, 1) * (out['m'] - out_uncond['m']))
                if return_j:
                    out['j'] = out_uncond['j'] + (y['scale'].view(-1, 1, 1, 1) * (out['j'] - out_uncond['j']))
        return out

