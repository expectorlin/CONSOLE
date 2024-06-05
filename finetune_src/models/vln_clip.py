import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class VLNCLIP(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        self.args = args

        self.clip_model = clip_model

        self.dtype = clip_model.dtype

        self.scoring_module = nn.Sequential(
            nn.Linear(512+768, 512),
            nn.ReLU(),
            nn.LayerNorm(512, eps=1e-12),
            nn.Dropout(0.1),
        )


    def forward(self, landmark, image_features, image_mask, mode, \
                ob_cand_lens, landmark_feature, inst=None):

        if mode == 'landmark_shift':
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # bs, len, dim
            landmark_feature = landmark_feature / landmark_feature.norm(dim=-1, keepdim=True)

            image_mask_new = image_mask.unsqueeze(-1).repeat(1, 1, image_features.size(-1))
            image_features.masked_fill_(image_mask_new != 1, 0)
            image_features_mean = image_features.mean(1)

            text_prob = 100.0 * torch.bmm(image_features_mean.unsqueeze(1),
                                          landmark_feature.permute(0, 2, 1).float()).squeeze(1)
            text_prob, landmark_ind = text_prob.max(-1)

            return landmark_ind

        elif mode == 'landmark_discovery':
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # bs, len, dim
            landmark_feature = landmark_feature / landmark_feature.norm(dim=-1, keepdim=True)

            image_mask_new = image_mask.unsqueeze(-1).repeat(1, 1, image_features.size(-1))
            image_features.masked_fill_(image_mask_new != 1, 0)
            image_features_mean = image_features.mean(1)

            inst_feature = self.scoring_module(torch.cat([inst,image_features_mean],dim=-1))
            inst_feature = inst_feature / inst_feature.norm(dim=-1, keepdim=True)
            cooccurrence_weight = 100.0 * torch.bmm(inst_feature.unsqueeze(1), landmark_feature.permute(0, 2, 1)).squeeze(1)

            candidate_prob = 100.0 * torch.bmm(image_features,
                                          landmark_feature.permute(0, 2, 1).float())

            image_mask_new = image_mask.unsqueeze(-1).repeat(1, 1, self.args.cooccurrence_num + 1)
            candidate_prob.masked_fill_(image_mask_new != 1, -float('inf'))
            candidate_prob_softmax = candidate_prob.softmax(1) # bs, candidate_num, self.args.cooccurrence_num+1


            corrected_landmark_feature = (landmark_feature.unsqueeze(1).repeat(1, image_features.size(1), 1, 1) * candidate_prob_softmax.unsqueeze(-1) \
                                                                                * cooccurrence_weight.unsqueeze(1).unsqueeze(-1).
                                                          repeat(1, image_features.size(1), 1, 1)).sum(2) # bs, candidate_num, 512


            corrected_candidate_prob = (candidate_prob_softmax * cooccurrence_weight.unsqueeze(1).repeat(1, image_features.size(1), 1)).sum(-1)
            corrected_candidate_prob.masked_fill_(image_mask != 1, -float('inf'))

            return corrected_landmark_feature, corrected_candidate_prob

        elif mode == 'text_cooccurrence':
            co_occurrence_list = []
            for bs_ind in range(len(landmark)):

                landmark_cooccu_list = landmark[bs_ind]['cooccu_landmark_enc']
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in landmark_cooccu_list]).cuda()
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_inputs)
                text_features = text_features.view(len(landmark[bs_ind]['landmark']), self.args.cooccurrence_num+1, -1)
                co_occurrence_list_item = text_features
                co_occurrence_list.append(co_occurrence_list_item)

            return co_occurrence_list

