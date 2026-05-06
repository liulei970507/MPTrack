"""
TBSI_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv, MLPv2, MLP
from lib.models.ostrack_quadra_zfuse_trans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq.vit_ostrack_care import vit_base_patch16_224_ostrack
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.tbsi_track.vit_tbsi_care import resize_pos_embed

class OSTrack(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, box_head_rgb, box_head_tir, aux_loss=False, head_type="CORNER", cls_head_rgb_reg=None, cls_head_tir_reg=None, cls_head_rgbt_reg=None, cls_head_rgb_cls=None, cls_head_tir_cls=None, cls_head_rgbt_cls=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.tbsi_fuse_search = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head
        self.box_head_rgb = box_head_rgb
        self.box_head_tir = box_head_tir
        
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
            self.box_head_rgb = _get_clones(self.box_head_rgb, 6)
            self.box_head_tir = _get_clones(self.box_head_tir, 6)
        
        self.cls_head_rgb_reg = cls_head_rgb_reg
        self.cls_head_tir_reg = cls_head_tir_reg
        self.cls_head_rgbt_reg = cls_head_rgbt_reg
        
        self.cls_head_rgb_cls = cls_head_rgb_cls
        self.cls_head_tir_cls = cls_head_tir_cls
        self.cls_head_rgbt_cls = cls_head_rgbt_cls

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                run_cls_head=False,
                run_all=False
                ):
        x_ori, x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        feat_last_ori = x_ori
        if isinstance(x, list):
            feat_last = x[-1]
            feat_last_ori = x_ori[-1]
        out = self.forward_head(feat_last_ori, feat_last, None, run_cls_head=run_cls_head, run_all=run_all)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature_ori, cat_feature, gt_score_map=None, run_cls_head=False, run_all=False):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        # rgbt feature
        enc_opt1 = cat_feature[:, num_template_token*3:num_template_token*3 + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]
        out_dict = {}
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.tbsi_fuse_search(opt_feat)

        # rgb feature
        enc_opt_rgb = cat_feature_ori[:, num_template_token*3:num_template_token*3 + num_search_token, :]
        opt1 = (enc_opt_rgb.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # tir feature 
        enc_opt_tir = cat_feature_ori[:, -num_search_token:, :]
        opt2 = (enc_opt_tir.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt2.size()
        opt_feat2 = opt2.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # rgbt
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out_dict.update({'pred_boxes': outputs_coord_new})
            out_dict.update({'score_map': score_map_ctr})
            out_dict.update({'size_map': size_map})
            out_dict.update({'offset_map': offset_map})
            
            # rgb
            # score_map_ctr, bbox, size_map, offset_map = self.box_head_rgb(opt_feat1, gt_score_map)
            # # outputs_coord = box_xyxy_to_cxcywh(bbox)
            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out_dict.update({'pred_boxes_rgb': outputs_coord_new})
            # out_dict.update({'score_map_rgb': score_map_ctr})
            # out_dict.update({'size_map_rgb': size_map})
            # out_dict.update({'offset_map_rgb': offset_map})
            # # tir
            # score_map_ctr, bbox, size_map, offset_map = self.box_head_tir(opt_feat2, gt_score_map)
            # # outputs_coord = box_xyxy_to_cxcywh(bbox)
            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out_dict.update({'pred_boxes_tir': outputs_coord_new})
            # out_dict.update({'score_map_tir': score_map_ctr})
            # out_dict.update({'size_map_tir': size_map})
            # out_dict.update({'offset_map_tir': offset_map})
            
            # return out
        else:
            raise NotImplementedError
        if True: # run_cls_head or run_all:
            # import pdb
            # pdb.set_trace()
            rgb_feat = enc_opt_rgb.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous().view(-1, 768, self.feat_sz_s, self.feat_sz_s)
            tir_feat = enc_opt_tir.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous().view(-1, 768, self.feat_sz_s, self.feat_sz_s)
            
            rgbt_feat = enc_opt.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous().view(-1, 768*2, self.feat_sz_s, self.feat_sz_s)
            
            out_dict.update({'reliablity_rgb_reg': self.cls_head_rgb_reg(rgb_feat, out_dict['pred_boxes']).sigmoid()})
            out_dict.update({'reliablity_tir_reg': self.cls_head_tir_reg(tir_feat, out_dict['pred_boxes']).sigmoid()})
            out_dict.update({'reliablity_rgbt_reg': self.cls_head_rgbt_reg(rgbt_feat, out_dict['pred_boxes']).sigmoid()})
            out_dict.update({'reliablity_rgbt_cls': self.cls_head_rgbt_cls(rgbt_feat)})
            out_dict.update({'reliablity_rgb_cls': self.cls_head_rgb_cls(rgb_feat)})
            out_dict.update({'reliablity_tir_cls': self.cls_head_tir_cls(tir_feat)})
            if not run_all:
                return out_dict
        return out_dict

def build_ostrack_quadra_zfuse_trans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ostrack':
        backbone = vit_base_patch16_224_ostrack(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                            tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH
                                            )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    box_head_rgb = build_box_head(cfg, hidden_dim)
    box_head_tir = build_box_head(cfg, hidden_dim)
    
    cls_head_rgb_reg = MLPv2(768+4, 256, 1, 3)
    cls_head_tir_reg = MLPv2(768+4, 256, 1, 3)
    cls_head_rgbt_reg = MLPv2(768*2+4, 256, 1, 3)
    
    cls_head_rgb_cls = MLP(768, 256, 1, 3)
    cls_head_tir_cls = MLP(768, 256, 1, 3)
    cls_head_rgbt_cls = MLP(768*2, 256, 1, 3)
    
    model = OSTrack(
        backbone,
        box_head,
        box_head_rgb,
        box_head_tir,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        cls_head_rgb_reg=cls_head_rgb_reg,
        cls_head_tir_reg=cls_head_tir_reg,
        cls_head_rgbt_reg=cls_head_rgbt_reg,
        cls_head_rgb_cls=cls_head_rgb_cls,
        cls_head_tir_cls=cls_head_tir_cls,
        cls_head_rgbt_cls=cls_head_rgbt_cls
    )
    
    if 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        # missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        # copy 一份 rgb 和 tir 的 head 参数
        # import pdb
        # pdb.set_trace()
        parm_dict = dict()
        for k,v in checkpoint["net"].items():
            parm_dict[k] = v
            if 'box_head' in k:
                parm_dict[k.replace('box_head', 'box_head_rgb')] = v
                parm_dict[k.replace('box_head', 'box_head_tir')] = v
        missing_keys, unexpected_keys = model.load_state_dict(parm_dict, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    elif 'DropTrack' in cfg.MODEL.PRETRAIN_FILE and training: 
        def pos_embed_filter(param):
            param['backbone.pos_embed_z'] = resize_pos_embed(param['backbone.pos_embed_z'], posemb_new=torch.zeros(1,64,768),num_tokens=0)
            param['backbone.pos_embed_x'] = resize_pos_embed(param['backbone.pos_embed_x'], posemb_new=torch.zeros(1,256,768),num_tokens=0)
            param['backbone.pos_embed_z'] += param['backbone.temporal_pos_embed_z']
            param['backbone.pos_embed_x'] += param['backbone.temporal_pos_embed_x']
            return param
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")
        checkpoint = pos_embed_filter(checkpoint["net"])
        parm_dict = dict()
        for k,v in checkpoint.items():
            parm_dict[k] = v
            if 'box_head' in k:
                parm_dict[k.replace('box_head', 'box_head_rgb')] = v
                parm_dict[k.replace('box_head', 'box_head_tir')] = v

        missing_keys, unexpected_keys = model.load_state_dict(parm_dict, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys',missing_keys, unexpected_keys)
    return model
