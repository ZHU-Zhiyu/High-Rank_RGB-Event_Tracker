import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed, PatchEmbed_event, xcorr_depthwise
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock
import random
import numpy as np

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed_event = PatchEmbed_event(in_chans=32, embed_dim=768, kernel_size=4, stride=4)
        # self.pos_embed_event = PatchEmbed_event(in_chans=32, embed_dim=768, kernel_size=4, stride=4)
        # self.pos_embed_event_z = PatchEmbed_event(in_chans=32, embed_dim=768, kernel_size=3, stride=1)
        # attn = CrossAttn(768, 4, 3072, 0.1, 'relu')
        # self.cross_attn = Iter_attn(attn, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
    def masking_fea(self,z, event_z, x, event_x, ratio=0.8 ):
        b,nz,c = z.shape
        b,nez,c = event_z.shape
        b,nx,c = x.shape
        b,nex,c = event_x.shape
        assert(nz == nez)
        assert(nx == nex)
        lenz_out = int(nz*ratio)
        lenx_out = int(nx*ratio)

        mask_nz = torch.rand(b,nz).float()
        mask_ez = torch.rand(b,nez).float()
        mask_nx = torch.rand(b,nx).float()
        mask_ex = torch.rand(b,nex).float()
        mask_nz = mask_nz>0.4
        mask_ez = mask_ez>0.4
        mask_ez = ~mask_nz + mask_ez
        mask_nz_idx = mask_nz.float().sort(1,descending=True)[-1].to(device = z.device)
        mask_ez_idx = mask_ez.float().sort(1,descending=True)[-1].to(device = z.device)
        mask_nx = mask_nx>0.4
        mask_ex = mask_ex>0.4
        mask_ex = ~mask_nx + mask_ex
        mask_nx_idx = mask_nx.float().sort(1,descending=True)[-1].to(device = z.device)
        mask_ex_idx = mask_ex.float().sort(1,descending=True)[-1].to(device = z.device)

        masked_z = torch.gather(z, 1, mask_nz_idx[:,:lenz_out,None].repeat([1,1,c]))
        masked_ez = torch.gather(event_z, 1, mask_ez_idx[:,:lenz_out,None].repeat([1,1,c]))
        masked_x = torch.gather(x, 1, mask_nx_idx[:,:lenx_out,None].repeat([1,1,c]))
        masked_ex = torch.gather(event_x, 1, mask_ex_idx[:,:lenx_out,None].repeat([1,1,c]))
        return masked_z, masked_ez, masked_x, masked_ex,{'x1':mask_nx_idx[:,:lenx_out],'x0':mask_nx_idx[:,lenx_out:],
                                                        'ex1':mask_ex_idx[:,:lenx_out],'ex0':mask_ex_idx[:,lenx_out:], }

    def forward_features(self, z, x, event_z, event_x,
                         mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,Track=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # print('shape of event_z before projection:{}, event_x:{}'.format(event_z.shape, event_x.shape))
        event_z = self.pos_embed_event(event_z)     # [:,:,:,:1000]
        event_x = self.pos_embed_event(event_x)     # B 768 1024
        x = self.patch_embed(x)
        z = self.patch_embed(z)
        # print('shape of event_z:{}, event_x:{}, x:{}, z:{}'.format(event_z.shape,event_x.shape,x.shape,z.shape ))
        event_z += self.pos_embed_z
        event_x += self.pos_embed_x
        z += self.pos_embed_z
        x += self.pos_embed_x

        # attention mask handling   # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
        if Track == False:
            z, event_z, x, event_x, token_idx = self.masking_fea(z, event_z, x, event_x, ratio=0.9) 
        x = combine_tokens(z, event_z, x, event_x, mode=self.cat_mode)        # 64+64+256+256=640
        # x = combine_tokens(z, x, event_z, event_x, mode=self.cat_mode)        # 64+64+256+256=640
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)
        # lens_z = self.pos_embed_z.shape[1]
        # lens_x = self.pos_embed_x.shape[1]
        lens_z = z.shape[1]
        lens_x = x.shape[1]
        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        out_attn = []
        for i, blk in enumerate(self.blocks):
            # out_global_s.append(global_index_s)
            # out_global_t.append(global_index_t)
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
            out_attn.append(attn)
        # print('shape of attn:{}, lens_z:{}, lens_x:{}'.format(attn.shape, lens_z, lens_x))

        
        out_attn_idx = random.choice(np.arange(len(out_attn)))
        out_attn = out_attn[out_attn_idx]
        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new*2]
        x = x[:, lens_z_new*2:]
        if Track == False:
            idx1 = token_idx['x1']
            idx0 = token_idx['x0']
            idex1 = token_idx['ex1']
            idex0 = token_idx['ex0']
            ex = x[:,idex1.shape[1]:]
            x = x[:,:idex1.shape[1]]
            # if removed_indexes_s and removed_indexes_s[0] is not None:
            #     removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = idx0.shape[1]
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([idx1, idx0], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
            ex = torch.cat([ex, pad_x], dim=1)
            index_all = torch.cat([idex1, idex0], dim=1)
            # recover original token order
            C = ex.shape[-1]
            ex = torch.zeros_like(ex).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=ex)
            x = torch.cat([x,ex],dim=1)
            x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        event_x = x[:, lens_x:]   # RGB head
        x = x[:, :lens_x]   # RGB head
        x = torch.cat([event_x, x], dim=1)
        
        aux_dict = {
            # "attn": attn,
            "attn": out_attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, event_z, event_x,
                ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,Track=False):

        x, aux_dict = self.forward_features(z, x, event_z, event_x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,Track=Track)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
