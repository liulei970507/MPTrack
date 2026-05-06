import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class RGBT210Dataset(BaseDataset):
    """ LasHeR dataset for RGB-T tracking.

    Publication:
        LasHeR: A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/abs/2104.13202

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.rgbt210_path # os.path.join(self.env_settings.gtot_path, split)
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path_i = '{}/{}/infrared'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/visible'.format(self.base_path, sequence_name)
        # frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".jpg")]
        # frame_list_i.sort(key=lambda f: int(f[1:-4]))
        # frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".jpg")]
        # frame_list_v.sort(key=lambda f: int(f[1:-4]))
        frame_list_v = sorted([p for p in os.listdir(frames_path_v) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frame_list_i = sorted([p for p in os.listdir(frames_path_i) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'rgbt210', ground_truth_rect.reshape(-1, 4))
    
    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        sequence_list= ['supbus', 'walkingwoman', 'glass', 'scooter', 'shake', 'raningcar', 'threewoman2', 'yellowcar', 'tallman', 'kite2', 'elecbike3', 'man3', 'car41', 'mandrivecar', 'glass2', 'floor-1', 'elecbike', 'manwithbag', 'playsoccer', 'womancross', 'inglassandmobile', 'guidepost', 'elecbikechange2', 'flower1', 'manonelecbike', 'supbus2', 'man7', 'run2', 'hotkettle', 'tricycle9', 'toy3', 'elecbike2', 'elecbikewithlight', 'walkingman20', 'dog', 'redcar2', 'woman4', 'blueCar', 'mancrossandup', 'tricycle6', 'toy4', 'bikemove1', 'womanleft', 'green', 'caraftertree', 'greyman', 'redmanchange', 'car20', 'womaninblackwithbike', 'redbag', 'threeman2', 'crossroad', 'whitesuv', 'man4', 'child', 'tricycletwo', 'manlight', 'toy1', 'woman48', 'rainingwaliking', 'soccer2', 'walkingtogether', 'greywoman', 'woman96', 'twoman', 'elecbikewithlight1', 'man55', 'afterrain', 'womanpink', 'soccer', 'mancross', 'womanfaraway', 'aftertree', 'people3', 'biketwo', 'woman100', 'tree3', 'tree5', 'straw', 'maningreen2', 'child4', 'luggage', 'mancross1', 'walkingman41', 'womanrun', 'man23', 'whitecarafterrain', 'stroller', 'nightcar', 'oldman', 'oldwoman', 'twowoman', 'man29', 'manypeople', 'twoman1', 'soccerinhand', 'man8', 'rmo', 'man5', 'baby', 'graycar2', 'walking40', 'bus6', 'man24', 'crouch', 'manonboundary', 'carLight', 'walkingwithbag2', 'woman6', 'man68', 'whitebag', 'carnotfar', 'people', 'manafterrain', 'tricycle', 'basketball2', 'man28', 'dog10', 'redcar', 'woamn46', 'diamond', 'womanred', 'children4', 'greentruck', 'child3', 'threeman', 'walkingtogether1', 'night2', 'man88', 'walkingtogetherright', 'fog6', 'tricyclefaraway', 'dog11', 'whitecar4', 'manontricycle', 'manwithbag4', 'manoccpart', 'children3', 'baketballwaliking', 'jump', 'nightthreepeople', 'run', 'woman', 'twoperson', 'elecbikeinfrontcar', 'children2', 'run1', 'whitecar3', 'nightrun', 'maninglass', 'bicyclecity', 'bikeman', 'woman89', 'shoeslight', 'car66', 'whitecar', 'call', 'carnotmove', 'bluebike', 'kettle', 'walkingman12', 'maninblack', 'manwithumbrella', 'tricycle2', 'maninred', 'walkingmantiny', 'manfaraway', 'manwithbasketball', 'manup', 'woman99', 'flower2', 'walkingman', 'elecbike10', 'man9', 'fog', 'twoelecbike1', 'manwithluggage', 'walkingwithbag1', 'manout2', 'notmove', 'man26', 'takeout', 'baginhand', 'bike', 'mobile', 'twoelecbike', 'hotglass', 'woamnwithbike', 'man45', 'elecbikewithhat', 'trees', 'car37', 'walkingman1', 'boundaryandfast', 'tree2', 'face1', 'womanwithbag6', 'push', 'man22', 'kite4', 'walking41', 'car10', 'man69', 'together', 'oldman2', 'people1', 'walkingnight', 'balancebike', 'car', 'carred']
        return sequence_list
