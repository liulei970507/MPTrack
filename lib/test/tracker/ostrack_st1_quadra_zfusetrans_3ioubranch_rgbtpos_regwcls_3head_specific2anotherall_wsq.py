import math

from lib.models.tbsi_track import build_tbsi_track
from lib.models.ostrack_quadra_zfuse_trans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq import build_ostrack_quadra_zfuse_trans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack_quadra_zfuse_trans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME][0]
        else:
            self.update_intervals = [50]
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME][0]
        else:
            self.update_threshold = [0.5]
        self.update_rule_ss = True
        self.update_rule_time = True
        print("Update interval is: ", self.update_intervals)
        print("Update threshold is: ", self.update_threshold)
        print("Update Rule SS:", self.update_rule_ss)
        print("Update Rule Time:", self.update_rule_time)
        self.xizhi_update = False
        if self.xizhi_update:
            self.update_intervals_tongbu = self.update_intervals[0]
            self.update_intervals_yibu_rgb = self.update_intervals[0]
            self.update_intervals_yibu_tir = self.update_intervals[0]
            self.update_intervals_gap = self.update_intervals[0]//5
            self.update_tongbu_idx = 0
            self.update_yibu_rgb_idx = 0
            self.update_yibu_tir_idx = 0
        self.update_idx = 0
        self.siglemodal_track = False
        self.update_metric_rm = False
        self.update_metric_rm_threshold = 0.5
        self.update_metric_regm = True
        self.update_metric_clsm = True
        self.state3update = False
        print('self.update_metric_rm,self.update_metric_regm,self.update_metric_clsm,self.update_metric_rm_threshold,self.state3update:',self.update_metric_rm,self.update_metric_regm,self.update_metric_clsm,self.update_metric_rm_threshold,self.state3update)
        # self.update_state = 0 # 0 配对 1 rgb 2 tir
        
    def initialize(self, image, info: dict):
        # initialize z_dict_list
        
        self.z_dict_list = []
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        with torch.no_grad():
            self.z_dict1 = template
            self.z_dict_list = []
            # 0 初始模板 1 同步模板 2 异步模板rgb 3 异步模板tir
            self.z_dict_list.append([template.tensors[:,:3,:,:], template.tensors[:,:3,:,:], template.tensors[:,:3,:,:], template.tensors[:,:3,:,:]])
            self.z_dict_list.append([template.tensors[:,-3:,:,:], template.tensors[:,-3:,:,:], template.tensors[:,-3:,:,:], template.tensors[:,-3:,:,:]])
            # 预测第一帧两个模态的可靠性
            x_patch_arr, _, x_amask_arr = sample_target(image, info['init_bbox'], self.params.search_factor,output_sz=self.params.search_size)  # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)
            search_img_v = search.tensors[:,:3,:,:]
            search_img_i = search.tensors[:,-3:,:,:]
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(template=self.z_dict_list, search=[search_img_v, search_img_i], ce_template_mask=self.box_mask_z, run_all=True)
            # reg model
            reliablity_rgb = out_dict['reliablity_rgb_reg'].view(-1).item() 
            reliablity_tir = out_dict['reliablity_tir_reg'].view(-1).item()
            reliablity_rgbt = out_dict['reliablity_rgbt_reg'].view(-1).item()
            # cls model
            reliablity_rgb_cls = out_dict['reliablity_rgb_cls'].sigmoid().view(-1).item() 
            reliablity_tir_cls = out_dict['reliablity_tir_cls'].sigmoid().view(-1).item()
            reliablity_rgbt_cls = out_dict['reliablity_rgbt_cls'].sigmoid().view(-1).item()
            updata_flag_rgb = True
            updata_flag_tir = True
            updata_flag_rgbt = True
            if self.update_metric_regm:
                updata_flag_rgb = updata_flag_rgb and reliablity_rgb >= self.update_threshold 
                updata_flag_tir = updata_flag_tir and reliablity_tir >= self.update_threshold
                updata_flag_rgbt = updata_flag_rgbt and reliablity_rgbt >= self.update_threshold 
            if self.update_metric_clsm:
                updata_flag_rgb = updata_flag_rgb and reliablity_rgb_cls>=0.5
                updata_flag_tir = updata_flag_tir and reliablity_tir_cls>=0.5
                updata_flag_rgbt = updata_flag_rgbt and reliablity_rgbt_cls>=0.5
            # response map
            if self.update_metric_rm:
                updata_flag_rgb = updata_flag_rgb and out_dict['score_map_rgb'].max().item()>=self.update_metric_rm_threshold
                updata_flag_tir = updata_flag_tir and out_dict['score_map_tir'].max().item()>=self.update_metric_rm_threshold
                updata_flag_rgbt = updata_flag_rgbt and out_dict['score_map'].max().item()>=self.update_metric_rm_threshold
            if updata_flag_rgbt and updata_flag_rgb and updata_flag_tir:
                self.update_state = 0
            elif updata_flag_rgbt and updata_flag_rgb:
                self.update_state = 1
            elif updata_flag_rgbt and updata_flag_tir:
                self.update_state = 2
            else:
                self.update_state = 3 
        
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        # print('self.frame_id, self.update_tongbu, self.update_yibu_rgb, self.update_yibu_tir, self.update_tongbu_idx, self.update_yibu_rgb_idx, self.update_yibu_tir_idx:', self.frame_id, self.update_tongbu, self.update_yibu_rgb, self.update_yibu_tir, self.update_tongbu_idx, self.update_yibu_rgb_idx, self.update_yibu_tir_idx)
        H, W, _ = image.shape
        self.frame_id += 1
        if self.xizhi_update:
            update_gap = min(self.frame_id - self.update_tongbu_idx, self.frame_id - max(self.update_yibu_rgb_idx, self.update_yibu_tir_idx))
        # print('self.frame_id, update_gap:', self.frame_id, update_gap)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            # x_dict = search
            # merge the template and the search
            # run the transformer
            # out_dict = self.network.forward(
            #     template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]], search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]], ce_template_mask=self.box_mask_z)
            search_img_v = search.tensors[:,:3,:,:]
            search_img_i = search.tensors[:,-3:,:,:]
            out_dict = self.network.forward(template=self.z_dict_list, search=[search_img_v, search_img_i], ce_template_mask=self.box_mask_z, run_all=True)
            # reg model
            reliablity_rgb = out_dict['reliablity_rgb_reg'].view(-1).item() 
            reliablity_tir = out_dict['reliablity_tir_reg'].view(-1).item()
            reliablity_rgbt = out_dict['reliablity_rgbt_reg'].view(-1).item()
            # cls model
            reliablity_rgb_cls = out_dict['reliablity_rgb_cls'].sigmoid().view(-1).item() 
            reliablity_tir_cls = out_dict['reliablity_tir_cls'].sigmoid().view(-1).item()
            reliablity_rgbt_cls = out_dict['reliablity_rgbt_cls'].sigmoid().view(-1).item()
            # print(self.frame_id, reliablity_rgb, reliablity_tir, reliablity_rgb_cls, reliablity_tir_cls)
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        if self.siglemodal_track:
            # rgb head
            pred_score_map_rgb = out_dict['score_map_rgb']
            response_rgb = self.output_window * pred_score_map_rgb
            pred_boxes_rgb = self.network.box_head.cal_bbox(response_rgb, out_dict['size_map_rgb'], out_dict['offset_map_rgb'])
            pred_boxes_rgb = pred_boxes_rgb.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box_rgb = (pred_boxes_rgb.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            state_rgb = clip_box(self.map_box_back(pred_box_rgb, resize_factor), H, W, margin=10)
            # tir head
            pred_score_map_tir = out_dict['score_map_tir']
            response_tir = self.output_window * pred_score_map_tir
            pred_boxes_tir = self.network.box_head.cal_bbox(response_tir, out_dict['size_map_tir'], out_dict['offset_map_tir'])
            pred_boxes_tir = pred_boxes_tir.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box_tir = (pred_boxes_tir.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            state_tir = clip_box(self.map_box_back(pred_box_tir, resize_factor), H, W, margin=10)
        
        # for update template
        if self.xizhi_update:
            if update_gap >= self.update_intervals_gap:
                if self.frame_id-self.update_tongbu_idx >= self.update_intervals_tongbu and reliablity_rgb > 0.5 and reliablity_tir > 0.5:
                    # print('self.update_tongbu, self.update_intervals_tongbu, reliablity_rgb, reliablity_tir:', self.update_tongbu, self.update_intervals_tongbu, reliablity_rgb, reliablity_tir)
                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                    self.z_dict_list[0][1] = template_t.tensors[:,:3,:,:]
                    self.z_dict_list[1][1] = template_t.tensors[:,-3:,:,:]
                    self.update_tongbu_idx = self.frame_id
                elif self.frame_id-self.update_yibu_rgb_idx >= self.update_intervals_yibu_rgb and reliablity_rgb > 0.5:
                    # print('self.update_yibu_gap, self.update_yibu_rgb, self.update_intervals_yibu_rgb, reliablity_rgb:', self.update_yibu_gap, self.update_yibu_rgb, self.update_intervals_yibu_rgb, reliablity_rgb)
                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                    self.z_dict_list[0][2] = template_t.tensors[:,:3,:,:]
                    self.update_yibu_rgb_idx = self.frame_id
                elif self.frame_id-self.update_yibu_tir_idx >= self.update_intervals_yibu_tir and reliablity_tir > 0.5:
                    # print('self.update_yibu_gap, self.update_yibu_tir, self.update_intervals_yibu_tir, reliablity_tir:', self.update_yibu_gap, self.update_yibu_tir, self.update_intervals_yibu_tir, reliablity_tir)
                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                    self.z_dict_list[1][3] = template_t.tensors[:,-3:,:,:]
                    self.update_yibu_tir_idx = self.frame_id
        else:
            updata_flag_rgb = True
            updata_flag_tir = True
            updata_flag_rgbt = True
            if self.update_metric_regm:
                updata_flag_rgb = updata_flag_rgb and reliablity_rgb >= self.update_threshold 
                updata_flag_tir = updata_flag_tir and reliablity_tir >= self.update_threshold
                updata_flag_rgbt = updata_flag_rgbt and reliablity_rgbt >= self.update_threshold 
            if self.update_metric_clsm:
                updata_flag_rgb = updata_flag_rgb and reliablity_rgb_cls>=0.5
                updata_flag_tir = updata_flag_tir and reliablity_tir_cls>=0.5
                updata_flag_rgbt = updata_flag_rgbt and reliablity_rgbt_cls>=0.5
            # response map
            if self.update_metric_rm:
                updata_flag_rgb = updata_flag_rgb and out_dict['score_map_rgb'].max().item()>=self.update_metric_rm_threshold
                updata_flag_tir = updata_flag_tir and out_dict['score_map_tir'].max().item()>=self.update_metric_rm_threshold
                updata_flag_rgbt = updata_flag_rgbt and out_dict['score_map'].max().item()>=self.update_metric_rm_threshold
            if updata_flag_rgbt and updata_flag_rgb and updata_flag_tir:
                state_new = 0
            elif updata_flag_rgbt and updata_flag_rgb:
                state_new = 1
            elif updata_flag_rgbt and updata_flag_tir:
                state_new = 2
            else:
                state_new = 3
            if self.update_rule_time and self.update_rule_ss:
                if state_new!=3:
                    if (self.frame_id - self.update_idx >= self.update_intervals and updata_flag_rgbt) or (state_new!=self.update_state): # v2
                        print('update, frame id:', self.frame_id, updata_flag_rgbt, updata_flag_rgb, updata_flag_tir)
                        if self.frame_id - self.update_idx >= self.update_intervals and updata_flag_rgbt: # v4 02
                            if state_new==0:
                                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                self.z_dict_list[0][1] = template_t.tensors[:,:3,:,:]
                                self.z_dict_list[1][1] = template_t.tensors[:,-3:,:,:]
                            elif state_new==1:
                                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                self.z_dict_list[0][2] = template_t.tensors[:,:3,:,:]
                            elif state_new==2:
                                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                self.z_dict_list[1][3] = template_t.tensors[:,-3:,:,:]
                            # else:
                            #     if self.state3update:
                            #         self.update_state = 3
                            #         self.update_idx = self.frame_id
                            self.update_idx = self.frame_id
                            self.update_state = state_new
                        else:
                            if self.update_state == 0:
                                if state_new==1:
                                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                    self.z_dict_list[0][2] = template_t.tensors[:,:3,:,:]
                                elif state_new==2:
                                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                    self.z_dict_list[1][3] = template_t.tensors[:,-3:,:,:]
                                # else:
                                #     if self.state3update:
                                #         self.update_state = 3
                                #         self.update_idx = self.frame_id
                                self.update_idx = self.frame_id
                                self.update_state = state_new
                            else:
                                if state_new==0:
                                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                    self.z_dict_list[0][1] = template_t.tensors[:,:3,:,:]
                                    self.z_dict_list[1][1] = template_t.tensors[:,-3:,:,:]
                                elif state_new==1:
                                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                    self.z_dict_list[0][2] = template_t.tensors[:,:3,:,:]
                                elif state_new==2:
                                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                                    self.z_dict_list[1][3] = template_t.tensors[:,-3:,:,:]
                                # else:
                                #     if self.state3update:
                                #         self.update_state = 3
                                #         self.update_idx = self.frame_id
                                self.update_idx = self.frame_id
                                self.update_state = state_new
                        
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            if self.siglemodal_track:
                return {"target_bbox": self.state, "target_bbox_rgb": state_rgb, "target_bbox_tir": state_tir, 'reliablity' : [reliablity_rgb, reliablity_tir]}
            else:
                return {"target_bbox": self.state, "target_bbox_rgb": self.state, "target_bbox_tir": self.state, 'reliablity' : [reliablity_rgb, reliablity_tir]}
            
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
