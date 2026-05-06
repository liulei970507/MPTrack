from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import random
import torch.nn.functional as F
class OSTrack3IoURegwCls2HeadActor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None, st='st1', run_cls_head=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.st = st
        self.run_cls_head = run_cls_head
        self.cls_num = 1
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'],st=self.st)

        return loss, status

    def forward_pass(self, data):
        # we need 3 template and 1 search region
        template_img_v = []
        template_img_i = []
        for i in range(self.settings.num_template):
            template_img_v.append(data['visible']['template_images'][i].view(-1, *data['visible']['template_images'].shape[2:]))  # (batch, 3, 128, 128)
            template_img_i.append(data['infrared']['template_images'][i].view(-1, *data['infrared']['template_images'].shape[2:]))  # (batch, 3, 128, 128)        
            
        search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)

        # 不做ce
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['visible']['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        if len(template_img_v) == 1:
            template_img_v = template_img_v[0]
            template_img_i = template_img_i[0]
        out_dict = self.net(template=[template_img_v, template_img_i],
                            search=[search_img_v, search_img_i],
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            # st2
                            run_cls_head=self.run_cls_head)

        return out_dict
    def consistency_loss(self, class_pred_rgb, iou_pred_rgb, 
                     class_pred_tir, iou_pred_tir, 
                     class_pred_rgbrt, iou_pred_rgbrt,
                     threshold_high=0.7, threshold_low=0.7):
        """
        计算分类结果与回归结果之间的一致性损失
        Args:
            class_pred_rgb (Tensor): RGB数据的分类预测，范围为[0, 1]，表示是否可靠
            iou_pred_rgb (Tensor): RGB数据的IoU回归预测
            class_pred_tir (Tensor): TIR数据的分类预测，范围为[0, 1]，表示是否可靠
            iou_pred_tir (Tensor): TIR数据的IoU回归预测
            class_pred_rgbrt (Tensor): RGBT数据的分类预测，范围为[0, 1]，表示是否可靠
            iou_pred_rgbrt (Tensor): RGBT数据的IoU回归预测
            threshold_high (float): IoU的上限阈值
            threshold_low (float): IoU的下限阈值
        Returns:
            loss (Tensor): 一致性损失
        """
        loss = 0

        # RGB分类与IoU之间的一致性约束
        loss += self.consistency_constraint(class_pred_rgb, iou_pred_rgb, threshold_high, threshold_low)
        # TIR分类与IoU之间的一致性约束
        loss += self.consistency_constraint(class_pred_tir, iou_pred_tir, threshold_high, threshold_low)
        # RGBT分类与IoU之间的一致性约束
        loss += self.consistency_constraint(class_pred_rgbrt, iou_pred_rgbrt, threshold_high, threshold_low)

        return loss

    def consistency_constraint(self, class_pred, iou_pred, threshold_high, threshold_low):
        """
        计算分类与回归结果之间的约束损失
        Args:
            class_pred (Tensor): 分类预测
            iou_pred (Tensor): IoU回归预测
            threshold_high (float): 高阈值
            threshold_low (float): 低阈值
        Returns:
            loss (Tensor): 该约束的损失
        """
        # 如果分类预测是可靠的，IoU应该大于threshold_high
        loss_reliable = F.relu(threshold_high - iou_pred) * (class_pred >= 0.5)
        
        # 如果分类预测是不可靠的，IoU应该小于threshold_high
        loss_unreliable = F.relu(iou_pred - threshold_low) * (class_pred < 0.5)
        
        # 将两个损失加起来
        return (loss_reliable + loss_unreliable).mean()

    def compute_losses(self, pred_dict, gt_dict, return_status=True, st='st1'):
        consistency_constraint = False
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes_rgb = pred_dict['pred_boxes_rgb']
        pred_boxes_tir = pred_dict['pred_boxes_tir']
        
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_rgb = box_cxcywh_to_xyxy(pred_boxes_rgb).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_tir = box_cxcywh_to_xyxy(pred_boxes_tir).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                        max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_rgb, iou_rgb = self.objective['giou'](pred_boxes_vec_rgb, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_tir, iou_tir = self.objective['giou'](pred_boxes_vec_tir, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_rgb, iou_rgb = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_tir, iou_tir = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        if st=='st1':
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            l1_loss_rgb = self.objective['l1'](pred_boxes_vec_rgb, gt_boxes_vec)  # (BN,4) (BN,4)
            l1_loss_tir = self.objective['l1'](pred_boxes_vec_tir, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
                location_loss_rgb = self.objective['focal'](pred_dict['score_map_rgb'], gt_gaussian_maps)
                location_loss_tir = self.objective['focal'](pred_dict['score_map_tir'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
                location_loss_rgb = torch.tensor(0.0, device=l1_loss.device)
                location_loss_tir = torch.tensor(0.0, device=l1_loss.device)
            # compute IoU predicted loss
            
            # reg
            if 'reliablity_rgb_reg' in pred_dict.keys():
                loss_ioubranch_rgb = self.objective['iou_branch'](pred_dict['reliablity_rgb_reg'].squeeze(-1), iou_rgb.detach())
                loss_ioubranch_tir = self.objective['iou_branch'](pred_dict['reliablity_tir_reg'].squeeze(-1), iou_tir.detach())
            else:
                loss_ioubranch_rgb = 0
                loss_ioubranch_tir = 0
            if 'reliablity_rgbt_reg' in pred_dict.keys():
                loss_ioubranch_rgbt = self.objective['iou_branch'](pred_dict['reliablity_rgbt_reg'].squeeze(-1), iou.detach())
            else:
                loss_ioubranch_rgbt = 0
            if 'reliablity_rgb_cls' in pred_dict.keys():
                binary_iou_rgb = (iou_rgb.detach() >= 0.7).float()
                binary_iou_tir = (iou_tir.detach() >= 0.7).float()
                loss_ioubranch_rgb_cls = self.objective['cls'](pred_dict['reliablity_rgb_cls'].squeeze(-1), binary_iou_rgb)
                loss_ioubranch_tir_cls = self.objective['cls'](pred_dict['reliablity_tir_cls'].squeeze(-1), binary_iou_tir)
                if 'reliablity_rgb_rm' in pred_dict.keys():
                    loss_ioubranch_rgb_rm = self.objective['cls'](pred_dict['reliablity_rgb_rm'].squeeze(-1), binary_iou_rgb)
                    loss_ioubranch_tir_rm = self.objective['cls'](pred_dict['reliablity_tir_rm'].squeeze(-1), binary_iou_tir)
                
                if 'reliablity_rgbt_cls' in pred_dict.keys():
                    binary_iou = (iou.detach() >= 0.7).float()
                    loss_ioubranch_rgbt_cls = self.objective['cls'](pred_dict['reliablity_rgbt_cls'].squeeze(-1), binary_iou)
                    if 'reliablity_rgbt_rm' in pred_dict.keys():
                        loss_ioubranch_rgbt_rm = self.objective['cls'](pred_dict['reliablity_rgbt_rm'].squeeze(-1), binary_iou)
                    else:
                        loss_ioubranch_rgbt_rm = 0.0
                else:
                    loss_ioubranch_rgbt_cls = 0.0
            # weighted sum
            # if 'reliablity_rgb_reg' in pred_dict.keys() and 'reliablity_rgb_cls' in pred_dict.keys():
            #     if 'reliablity_rgbt_reg' not in pred_dict.keys():
            #         loss =  self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir + loss_ioubranch_rgb + loss_ioubranch_tir) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir + loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls)
            #     else:
            #         loss =  self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir + loss_ioubranch_rgb + loss_ioubranch_tir + loss_ioubranch_rgbt) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir + loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls + loss_ioubranch_rgbt_cls)
                
            # elif 'reliablity_rgb_reg' in pred_dict.keys():
            #     loss =  self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir + loss_ioubranch_rgb + loss_ioubranch_tir) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir)
            # elif 'reliablity_rgb_cls' in pred_dict.keys():
            #     loss =  self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir + loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls)
            # else:
            #     loss =  self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir)
            
            
            loss =  self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir)
            # default 
            # if 'reliablity_rgb_cls' in pred_dict.keys():
            #     loss = loss + self.loss_weight['focal'] * (loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls + loss_ioubranch_rgbt_cls)
            # if 'reliablity_rgb_reg' in pred_dict.keys():
            #     loss = loss + self.loss_weight['l1'] * (loss_ioubranch_rgb + loss_ioubranch_tir + loss_ioubranch_rgbt)
            # if 'reliablity_rgb_rm' in pred_dict.keys():
            #     loss = loss + self.loss_weight['focal'] * (loss_ioubranch_rgb_rm + loss_ioubranch_tir_rm + loss_ioubranch_rgbt_rm)
            # wmutl0p1
            if 'reliablity_rgb_cls' in pred_dict.keys():
                loss = loss + 0.1*self.loss_weight['focal'] * (loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls + loss_ioubranch_rgbt_cls)
            if 'reliablity_rgb_reg' in pred_dict.keys():
                loss = loss + 0.1*self.loss_weight['l1'] * (loss_ioubranch_rgb + loss_ioubranch_tir + loss_ioubranch_rgbt)
            if 'reliablity_rgb_rm' in pred_dict.keys():
                loss = loss + 0.1*self.loss_weight['focal'] * (loss_ioubranch_rgb_rm + loss_ioubranch_tir_rm + loss_ioubranch_rgbt_rm)
            # 0p1
            # if 'reliablity_rgb_cls' in pred_dict.keys():
            #     loss = loss + 0.1 * (loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls + loss_ioubranch_rgbt_cls)
            # if 'reliablity_rgb_reg' in pred_dict.keys():
            #     loss = loss + 0.1 * (loss_ioubranch_rgb + loss_ioubranch_tir + loss_ioubranch_rgbt)
            # if 'reliablity_rgb_rm' in pred_dict.keys():
            #     loss = loss + 0.1 * (loss_ioubranch_rgb_rm + loss_ioubranch_tir_rm + loss_ioubranch_rgbt_rm)
            
            if consistency_constraint:
                # import pdb
                # pdb.set_trace()
                loss_consistency_constraint = self.consistency_loss(pred_dict['reliablity_rgb_cls'].sigmoid(),pred_dict['reliablity_rgb_reg'],
                                             pred_dict['reliablity_tir_cls'].sigmoid(),pred_dict['reliablity_tir_reg'],
                                             pred_dict['reliablity_rgbt_cls'].sigmoid(),pred_dict['reliablity_rgbt_reg'])
            else:
                loss_consistency_constraint = 0.
            loss = loss + loss_consistency_constraint
            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                mean_iou_rgb = iou_rgb.detach().mean()
                mean_iou_tir = iou_tir.detach().mean()
                status = {"Loss/total": loss.item(),
                        "Loss/giou": giou_loss.item(),
                        "Loss/giou_rgb": giou_loss_rgb.item(),
                        "Loss/giou_tir": giou_loss_tir.item(),
                        "Loss/l1": l1_loss.item(),
                        "Loss/l1_rgb": l1_loss_rgb.item(),
                        "Loss/l1_tir": l1_loss_tir.item(),
                        "Loss/location": location_loss.item(),
                        "Loss/location_rgb": location_loss_rgb.item(),
                        "Loss/location_tir": location_loss_tir.item(),
                        "IoU": mean_iou.item(),
                        "IoU_rgb": mean_iou_rgb.item(),
                        "IoU_tir": mean_iou_tir.item(),
                        "IoU_branch_rgb_reg": loss_ioubranch_rgb.item() if 'reliablity_rgb_reg' in pred_dict.keys() else 0.,
                        "IoU_branch_tir_reg": loss_ioubranch_tir.item() if 'reliablity_rgb_reg' in pred_dict.keys() else 0.,
                        "IoU_branch_rgbt_reg": loss_ioubranch_rgbt.item() if 'reliablity_rgbt_reg' in pred_dict.keys() else 0.,
                        "IoU_branch_rgb_cls": loss_ioubranch_rgb_cls.item() if 'reliablity_rgb_cls' in pred_dict.keys() else 0.,
                        "IoU_branch_tir_cls": loss_ioubranch_tir_cls.item() if 'reliablity_rgb_cls' in pred_dict.keys() else 0.,
                        "IoU_branch_rgbt_cls": loss_ioubranch_rgbt_cls.item() if 'reliablity_rgbt_cls' in pred_dict.keys() else 0.,
                        "IoU_branch_rgb_rm": loss_ioubranch_rgb_rm.item() if 'reliablity_rgb_rm' in pred_dict.keys() else 0.,
                        "IoU_branch_tir_rm": loss_ioubranch_tir_rm.item() if 'reliablity_rgb_rm' in pred_dict.keys() else 0.,
                        "IoU_branch_rgbt_rm": loss_ioubranch_rgbt_rm.item() if 'reliablity_rgbt_rm' in pred_dict.keys() else 0.,
                        'loss_consistency_constraint': loss_consistency_constraint.item() if consistency_constraint else 0.
                        }
                return loss, status
            else:
                return loss
        elif st=='st2':
            if 'reliablity_rgb_reg' in pred_dict.keys():
                loss_ioubranch_rgb = self.objective['iou_branch'](pred_dict['reliablity_rgb_reg'].squeeze(-1), iou_rgb.detach())
                loss_ioubranch_tir = self.objective['iou_branch'](pred_dict['reliablity_tir_reg'].squeeze(-1), iou_tir.detach())
            
            if 'reliablity_rgb_cls' in pred_dict.keys():
                binary_iou_rgb = (iou_rgb.detach() >= 0.7).float()
                binary_iou_tir = (iou_tir.detach() >= 0.7).float()
                loss_ioubranch_rgb_cls = self.objective['cls'](pred_dict['reliablity_rgb_cls'].squeeze(-1), binary_iou_rgb)
                loss_ioubranch_tir_cls = self.objective['cls'](pred_dict['reliablity_tir_cls'].squeeze(-1), binary_iou_tir)
            # weighted sum
            if 'reliablity_rgb_reg' in pred_dict.keys() and 'reliablity_rgb_cls' in pred_dict.keys():
                loss =  self.loss_weight['l1'] * (loss_ioubranch_rgb + loss_ioubranch_tir) + self.loss_weight['focal'] * (loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls)
            elif 'reliablity_rgb_reg' in pred_dict.keys():
                loss =  self.loss_weight['l1'] * (loss_ioubranch_rgb + loss_ioubranch_tir)
            elif 'reliablity_rgb_cls' in pred_dict.keys():
                loss =  self.loss_weight['focal'] * (loss_ioubranch_rgb_cls + loss_ioubranch_tir_cls)
            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                mean_iou_rgb = iou_rgb.detach().mean()
                mean_iou_tir = iou_tir.detach().mean()
                status = {"Loss/total": loss.item(),
                        "IoU": mean_iou.item(),
                        "IoU_rgb": mean_iou_rgb.item(),
                        "IoU_tir": mean_iou_tir.item(),
                        "IoU_branch_rgb_reg": loss_ioubranch_rgb.item() if 'reliablity_rgb_reg' in pred_dict.keys() else 0.,
                        "IoU_branch_tir_reg": loss_ioubranch_tir.item() if 'reliablity_rgb_reg' in pred_dict.keys() else 0.,
                        "IoU_branch_rgb_cls": loss_ioubranch_rgb_cls.item() if 'reliablity_rgb_cls' in pred_dict.keys() else 0.,
                        "IoU_branch_tir_cls": loss_ioubranch_tir_cls.item() if 'reliablity_rgb_cls' in pred_dict.keys() else 0.}
                return loss, status
            else:
                return loss