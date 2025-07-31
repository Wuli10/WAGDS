import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict
from pkg_resources import packaging
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Any, Union, List
from models.DSAG import DSAG

_tokenizer = _Tokenizer()

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class PromptLearner(nn.Module):
    def __init__(self, classnames, ln_final, token_embedding):
        super().__init__()
        n_cls = len(classnames)  # 36
        n_ctx = 8
        ctx_init = ""
        dtype = ln_final.weight.dtype
        ctx_dim = ln_final.weight.shape[0]   # 512

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)  # 8 learnable prompt

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # print(name_lens)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS   

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class AttentionPool2d(nn.Module):  # 16, 384, 64, 1024
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):   #(b,384,16,16)
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC   # (256,b,384)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  # (257,b,384)
        x_1 = x + self.positional_embedding[:, None, :].to(x.dtype)   # (257,b,384)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x[0], x[1:], att[:, 1:, 1:]
    

    
class Model(nn.Module):
    def __init__(self, args, embed_dim:int, context_length: int, vocab_size: int, 
                 transformer_width: int, transformer_heads: int, 
                 transformer_layers: int, num_classes=36, pretrained=True, n=3, D=512, dino_pretrained='dinov2_vits14', n_layer=12, pred_decoder_args={"mlp_dim":1024, "depth":2, "use_up":2, "use_additional_token":True}):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.pretrained = pretrained
        self.dino_pretrained = dino_pretrained
        self.n = n
        self.D = D
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self.context_length = context_length
        if args.divide == "Seen":
            self.classnames = ['beat', "boxing", "brush_with", "carry", "catch",
                         "cut", "cut_with", "drag", 'drink_with', "eat",
                         "hit", "hold", "jump", "kick", "lie_on", "lift",
                         "look_out", "open", "pack", "peel", "pick_up",
                         "pour", "push", "ride", "sip", "sit_on", "stick",
                         "stir", "swing", "take_photo", "talk_on", "text_on",
                         "throw", "type_on", "wash", "write"]
                         
        elif args.divide=="Unseen":
            self.classnames = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
        else: # HICO-IIF
            self.classnames = ['cut_with', 'drink_with', 'hold', 'open', 'pour', 'sip', 'stick', 'stir', 'swing', 'type_on']
        
        # dino-v2
        self.dino_model = torch.hub.load('./dinov2', self.dino_pretrained, source='local').cuda()
        self.vit_feat_dim = int(self.dino_model.norm.weight.shape[0])
        self.stride = 16
        self.patch = 16
        
        
        

        self.aff_proj = Mlp(in_features=int(self.vit_feat_dim*self.n_layer), hidden_features=int(self.vit_feat_dim*2), out_features=self.vit_feat_dim,
                            act_layer=nn.GELU, drop=0.)
        
        self.aff_ego_proj = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        ) for _ in range(self.n)])
        self.aff_exo_proj = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        ) for _ in range(self.n)])

        
        
        self.logit_scale_ego = nn.ParameterList([nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) for _ in range(self.n)])
        self.logit_scale_exo = nn.ParameterList([nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) for _ in range(self.n)])
        self.attnpool = nn.ModuleList([AttentionPool2d(16, self.vit_feat_dim, 64, embed_dim) for _ in range(self.n)])
        self.attnpool_exo = nn.ModuleList([AttentionPool2d(16, self.vit_feat_dim, 64, embed_dim) for _ in range(self.n)])
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )


        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.prompt_learner = PromptLearner(self.classnames, self.ln_final, self.token_embedding)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.fc_ego = nn.ModuleList([nn.Linear(self.vit_feat_dim, self.num_classes) for _ in range(self.n)])
        self.fc_exo = nn.ModuleList([nn.Linear(self.vit_feat_dim, self.num_classes) for _ in range(self.n)])
        self.fc = nn.ModuleList([nn.Linear(self.vit_feat_dim, self.num_classes) for _ in range(self.n)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.weight_ego = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.n)])
        self.weight_exo = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.n)])

        self.weight_trans = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(self.n)])



        self.DSAG_list = nn.ModuleList([DSAG(
            self.logit_scale_ego[i], self.logit_scale_exo[i], 
            self.weight_ego[i], self.weight_exo[i], 
            1, self.num_classes, self.embed_dim,
            self.fc_ego[i], self.fc_ego[i], self.avgpool, 
            self.attnpool[i], self.attnpool_exo[i],
            self.aff_exo_proj[i], self.aff_ego_proj[i],
            self.batch_minmax_normalize) for i in range(self.n)])


    
    def encode_text(self, per, text):
        x = per + self.token_embedding(text.cuda()).float()  # [num_class, transformer_width, d_model] 
        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)  # NLD -> LND   
        x = self.transformer(x)    
        x = x.permute(1, 0, 2)  # LND -> NLD   
        x = self.ln_final(x).float()   
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x

    
    def forward(self, exocentric, egocentric_image, label, text):
        target = label.long().squeeze()  # b
        
        b, n, c, h, w = exocentric.size()
        exocentric_input = exocentric.view(b * n, c, h, w) # b*3, 3, 224, 224
        
        
        # dino_vit
        
        dino_ego_out = self.dino_model.get_intermediate_layers(egocentric_image, n=self.n_layer, return_class_token=True)
        dino_exo_out = self.dino_model.get_intermediate_layers(exocentric_input, n=self.n_layer, return_class_token=True)

        ego_desc = dino_ego_out[0][0]  # penultimate: (b,256,384)
        exo_desc = dino_exo_out[0][0]  # penultimate: (b*3,256,384) 
        for i in range(1, len(dino_ego_out)):
            ego_desc = torch.cat((ego_desc, dino_ego_out[i][0]), dim=2)  # (b,256,384*n_layer)
            exo_desc = torch.cat((exo_desc, dino_exo_out[i][0]), dim=2)  # (b*3,256,384*n_layer)
            
    
        
        ego_proj = self.aff_proj(ego_desc)   # (b,256,384*n_layer)->(b,256,384)
        exo_proj = self.aff_proj(exo_desc)   # (b*3,256,384*n_layer)->(b*3,256,384)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)   # (b,384,16,16)
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)  # (b*3,384,16,16)



        # text branch
        prompts = self.prompt_learner()   # (36,77,512)
        
        
        tokenized_prompts = self.tokenized_prompts   # (36/nujm_class, 77/context_length)
        
        text_features = self.encode_text(prompts, tokenized_prompts)   # (36/nujm_class, 1024/embed_dim)
       

        e_b = ego_proj.shape[0]


        text_f_dec = torch.ones((e_b, self.embed_dim)).cuda()   #(b,1024)
        for i in range(e_b):
            text_f_dec[i] = text_features[label[i]]


        exo_proj = exo_proj.view(b, n, -1, exo_proj.shape[-1], exo_proj.shape[-1])   # (b,3,384,16,16)

        exo_score_list = []
        ego_score_list = []
        logits_per_text_list = []
        logits_per_image_list = []
        logits_per_text_exo_list = []
        logits_per_image_exo_list = []
        sim_loss_list = []
        cam1_list = []
        cam_list = []
        egocentric_branch_list = []
        mu_att_list = []

        ego_proj_pre = ego_proj
        
        for i in range(n):
            exo_proj_i = exo_proj[:, i, :, :, :]   # (b,384,16,16)
            b, _, h, w = exo_proj_i.shape

            exo_score, ego_score, logits_per_text, logits_per_image, logits_per_text_exo, logits_per_image_exo, sim_loss, cam, egocentric_branch, mu_att = self.DSAG_list[i](
                exo_proj_i, ego_proj, label, text_features
            )
            
            exo_score_list.append(exo_score)
            ego_score_list.append(ego_score)
            logits_per_text_list.append(logits_per_text)
            logits_per_image_list.append(logits_per_image)
            logits_per_text_exo_list.append(logits_per_text_exo)
            sim_loss_list.append(sim_loss)
            logits_per_image_exo_list.append(logits_per_image_exo)
            mu_att_list.append(mu_att)

            cam1 = mu_att_list[0] @ (cam.view(b, -1, 1))
            cam1 = cam1.view(b, h, w)
            cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            cam1 = F.interpolate(cam1.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            cam = self.batch_minmax_normalize(cam)
            cam1 = self.batch_minmax_normalize(cam1)
            cam1_list.append(cam1)
            cam_list.append(cam)
            

            ego_proj = self.weight_trans[i] * ego_proj_pre + (1-self.weight_trans[i]) * egocentric_branch


        return exo_score_list, ego_score_list, logits_per_text_list, logits_per_image_list, logits_per_text_exo_list, logits_per_image_exo_list, sim_loss_list, cam1_list, cam_list


    
    @torch.no_grad()
    def get(self, egocentric_image, label, text):
        
    
        dino_ego_out = self.dino_model.get_intermediate_layers(egocentric_image, n=self.n_layer, return_class_token=True)
        ego_desc = dino_ego_out[0][0].detach()  # penultimate: (b,256,384)
        for i in range(1, len(dino_ego_out)):
            ego_desc = torch.cat((ego_desc, dino_ego_out[i][0].detach()), dim=2)  # (b,256,384*2)


        ego_proj = self.aff_proj(ego_desc)   # (b,256,384)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        
        
        # text branch
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.encode_text(prompts, tokenized_prompts)

        e_b = ego_proj.shape[0]

        text_f_dec = torch.ones((e_b, self.embed_dim)).cuda()   #(b,1024)
        for i in range(e_b):
            text_f_dec[i] = text_features[label[i]]

        cam_list = []
        cam1_list = []
        mu_att_list = []

        ego_proj_pre = ego_proj


        for i in range(self.n):
            b, _, h, w = ego_proj.shape
            
            cam, egocentric_branch, mu_att = self.DSAG_list[i](
                None, ego_proj, label, text_features
            )
            mu_att_list.append(mu_att)
            cam1 = mu_att_list[0] @ (cam.view(b, -1, 1))
            cam1 = cam1.view(b, h, w)
            cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            cam1 = F.interpolate(cam1.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
            cam = self.batch_minmax_normalize(cam)
            cam1 = self.batch_minmax_normalize(cam1)

            cam_list.append(cam)
            cam1_list.append(cam1)
            ego_proj = self.weight_trans[i] * ego_proj_pre + (1-self.weight_trans[i]) * egocentric_branch


        
        return cam_list, cam1_list
    

    def batch_minmax_normalize(self, x):
    
        min_vals = x.view(x.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        max_vals = x.view(x.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)

        
        normalized_x = (x - min_vals) / (max_vals - min_vals + 1e-8) 

        return normalized_x
    
    def _reshape_transform(self, tensor, patch_size, stride):
        height = (tensor.shape[1] - patch_size) // stride + 1
        width = (tensor.shape[1] - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))   # (b,16,16,384)
        result = result.transpose(2, 3).transpose(1, 2).contiguous()   # (b,384,16,16)
        return result
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens

        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

def MODEL(args, num_classes=36,
          pretrained=True, n=3, D=512, n_layer=12):
    dict = 'RN50.pt'
    state_dict =  torch.jit.load(dict)
    state_dict = state_dict.state_dict()
    
    embed_dim = state_dict["text_projection"].shape[1]     # 1024
    context_length = state_dict["positional_embedding"].shape[0]   # 77
    vocab_size = state_dict["token_embedding.weight"].shape[0]   # 49408
    transformer_width = state_dict["ln_final.weight"].shape[0]    # 512
    transformer_heads = transformer_width // 64     # 8
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))   

    model = Model(args, embed_dim=embed_dim, context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width,
                  transformer_heads=transformer_heads,transformer_layers=transformer_layers, num_classes=num_classes, pretrained=pretrained, n=n, D=D, n_layer=n_layer)

    model_dict = model.state_dict()

    par = []
    par_no = []
    pretrained_dict = {}
    for para in model.named_parameters():
        k = para[0]
        if k in state_dict or 'dino' in k:
            par.append(para[0])     # name of the model's parameters in clip's pretrained model 
        else:
            par_no.append(para[0])
    for k, v in state_dict.items(): 
        if k in model_dict:
            pretrained_dict[k] = v   # clip's pretrained model weights for model
    model_dict.update(pretrained_dict)    
    model.load_state_dict(model_dict)


    for n, m in model.named_parameters():
        if par_no:
            if isinstance(par_no, str):
                if not par_no in n:
                    m.requires_grad = False
            elif isinstance(par_no, list):
                count = 0
                for i in range(len(par_no)):
                    i_layer = str(par_no[i])
                    if i_layer == n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count > 0:
                    print('Finetune layer in backbone:', n)
            else:
                assert AttributeError("Dont support the type of par_no!")
        else:
            m.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} requires grad: {param.requires_grad}")

    return model, par