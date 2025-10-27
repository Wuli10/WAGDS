import torch
import torch.nn as nn
from torch.nn import functional as F

class DSAG(nn.Module):
    def __init__(self, logit_scale_ego, logit_scale_exo, weight_ego, weight_exo, n, num_classes, embed_dim, fc, fc_exo, avgpool, attnpool, attnpool_exo, aff_exo_proj, aff_ego_proj, batch_minmax_normalize):
        super(DSAG, self).__init__()
        self.logit_scale = logit_scale_ego
        self.logit_scale_exo = logit_scale_exo
        self.weight_ego = weight_ego
        self.weight_exo = weight_exo
        self.n = n
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fc = fc
        self.fc_exo = fc_exo
        self.avgpool = avgpool
        self.attnpool = attnpool
        self.attnpool_exo = attnpool_exo
        self.aff_exo_proj = aff_exo_proj
        self.aff_ego_proj = aff_ego_proj
        self.batch_minmax_normalize = batch_minmax_normalize

    def forward(self, exo_proj, ego_proj, label, text_features):

        if exo_proj is not None:
            b_exo, channel_exo, h_exo, w_exo = exo_proj.shape
            pre_exo = exo_proj
            image_features_exo, pre_exo, mu_att_exo = self.attnpool_exo(pre_exo)

            image_features_exo = F.normalize(image_features_exo, dim=1, p=2)
            text_features_exo = text_features
            text_features_exo = F.normalize(text_features_exo, dim=1, p=2)

            logit_scale_exo = self.logit_scale_exo.exp()
            logits_per_image_exo = logit_scale_exo * image_features_exo @ text_features_exo.t()
            logits_per_text_exo = logits_per_image_exo.t()

            text_f_exo = torch.ones((b_exo//self.n, self.embed_dim)).cuda()
            for i in range(b_exo//self.n):
                text_f_exo[i] = text_features_exo[label[i]]

            text_f_exo = text_f_exo.repeat(self.n, 1)
            att_exoproj = F.normalize(pre_exo, dim=2, p=2)
            attexo = logit_scale_exo * att_exoproj.permute(1, 0, 2) @ text_f_exo.unsqueeze(2)
            attexo = torch.sigmoid(F.normalize(attexo, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, channel_exo)
            exo_proj = self.weight_exo * attexo.permute(1, 2, 0).view(b_exo, channel_exo, h_exo, w_exo) * exo_proj + (1 - self.weight_exo) * exo_proj

            e_b, e_c, e_h, e_w = ego_proj.shape
            pre_ego = ego_proj
            image_features, ego_proj, mu_att = self.attnpool(ego_proj)

            mu_att = mu_att / torch.sum(mu_att, dim=1, keepdim=True)
            mu_att = mu_att / torch.sum(mu_att, dim=2, keepdim=True)
            for _ in range(2):
                mu_att = mu_att / torch.sum(mu_att, dim=1, keepdim=True)
                mu_att = mu_att / torch.sum(mu_att, dim=2, keepdim=True)
            mu_att = (mu_att + mu_att.permute(0, 2, 1)) / 2
            mu_att = torch.matmul(mu_att, mu_att)

            image_features = F.normalize(image_features, dim=1, p=2)
            text_features = F.normalize(text_features, dim=1, p=2)

            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            text_f = torch.ones((e_b, self.embed_dim)).cuda()
            for i in range(e_b):
                text_f[i] = text_features[label[i]]
            att_egoproj = F.normalize(ego_proj, dim=2, p=2)
            attego = logit_scale * att_egoproj.permute(1, 0, 2) @ text_f.unsqueeze(2)
            attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, e_c)

            attn_expand = attego.permute(1, 0, 2).repeat(self.n, 1, 1).reshape(b_exo, -1)
            sim_loss = torch.max(
                1 - F.cosine_similarity(attn_expand, attexo.permute(1, 0, 2).reshape(b_exo, -1), dim=1),
                torch.zeros(attexo.shape[1]).to(attexo.device)
            )

            ego_proj = self.weight_ego * attego.permute(1, 2, 0).view(e_b, e_c, e_h, e_w) * pre_ego + (1 - self.weight_ego) * pre_ego

            exocentric_branch = self.aff_exo_proj(exo_proj)
            egocentric_branch = self.aff_ego_proj(ego_proj)

            exo_pool = self.avgpool(exocentric_branch)
            exo_pool = exo_pool.view(exo_pool.size(0), -1)
            exo_score = self.fc_exo(exo_pool)

            batch, channel, h, w = exocentric_branch.shape
            exocentric_branch = exocentric_branch.view(batch // self.n, self.n, channel, h, w).mean(1)
            batch = batch // self.n

            target = label.long().squeeze()
            exo_weight = self.fc_exo.weight[target]
            exo_weight = exo_weight.view(batch, channel, 1, 1).expand_as(exocentric_branch)
            exo_feature = (exo_weight * exocentric_branch)


            ego_pool = self.avgpool(egocentric_branch)
            ego_pool = ego_pool.view(ego_pool.size(0), -1)
            ego_score = self.fc(ego_pool)

            ego_weight = self.fc.weight[target]
            ego_weight = ego_weight.view(batch, channel, 1, 1).expand_as(egocentric_branch)
            ego_feature = (ego_weight * egocentric_branch)

            cam = ego_feature.mean(1)
            cam = cam.view(batch, h, w)


            
            return exo_score, ego_score, logits_per_text, logits_per_image, logits_per_text_exo, logits_per_image_exo, sim_loss, cam, egocentric_branch, mu_att
        else:
            
            e_b, e_c, e_h, e_w = ego_proj.shape
            pre_ego = ego_proj
            image_features, ego_proj, mu_att = self.attnpool(ego_proj)

            mu_att = mu_att / torch.sum(mu_att, dim=1, keepdim=True)
            mu_att = mu_att / torch.sum(mu_att, dim=2, keepdim=True)
            for _ in range(2):
                mu_att = mu_att / torch.sum(mu_att, dim=1, keepdim=True)
                mu_att = mu_att / torch.sum(mu_att, dim=2, keepdim=True)
            mu_att = (mu_att + mu_att.permute(0, 2, 1)) / 2
            mu_att = torch.matmul(mu_att, mu_att)

            image_features = F.normalize(image_features, dim=1, p=2)
            text_features = F.normalize(text_features, dim=1, p=2)

            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            text_f = torch.ones((e_b, self.embed_dim)).cuda()
            for i in range(e_b):
                text_f[i] = text_features[label[i]]
            att_egoproj = F.normalize(ego_proj, dim=2, p=2)
            attego = logit_scale * att_egoproj.permute(1, 0, 2) @ text_f.unsqueeze(2)
            attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, e_c)


            ego_proj = self.weight_ego * attego.permute(1, 2, 0).view(e_b, e_c, e_h, e_w) * pre_ego + (1 - self.weight_ego) * pre_ego

            
            egocentric_branch = self.aff_ego_proj(ego_proj)
            batch, channel, h, w = egocentric_branch.shape


            ego_pool = self.avgpool(egocentric_branch)
            ego_pool = ego_pool.view(ego_pool.size(0), -1)
            ego_score = self.fc(ego_pool)

            target = label.long().squeeze()

            ego_weight = self.fc.weight[target]
            ego_weight = ego_weight.view(batch, channel, 1, 1).expand_as(egocentric_branch)
            ego_feature = (ego_weight * egocentric_branch)

            cam = ego_feature.mean(1)
            cam = cam.view(batch, h, w)


            return cam, egocentric_branch, mu_att