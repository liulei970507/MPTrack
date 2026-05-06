class EnvironmentSettings:
    def __init__(self):
        import os
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.workspace_dir = root_dir    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(root_dir, 'tensorboard')    # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(root_dir, 'pretrained_networks')
        self.lasot_dir = '/mnt/dataShare/Teacher/liulei/Dataset//lasot'
        self.got10k_dir = '/mnt/dataShare/Teacher/liulei/Dataset//got10k/train'
        self.got10k_val_dir = '/mnt/dataShare/Teacher/liulei/Dataset//got10k/val'
        self.lasot_lmdb_dir = '/mnt/dataShare/Teacher/liulei/Dataset//lasot_lmdb'
        self.got10k_lmdb_dir = '/mnt/dataShare/Teacher/liulei/Dataset//got10k_lmdb'
        self.trackingnet_dir = '/mnt/dataShare/Teacher/liulei/Dataset//trackingnet'
        self.trackingnet_lmdb_dir = '/mnt/dataShare/Teacher/liulei/Dataset//trackingnet_lmdb'
        self.coco_dir = '/mnt/dataShare/Teacher/liulei/Dataset//coco'
        self.coco_lmdb_dir = '/mnt/dataShare/Teacher/liulei/Dataset//coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/dataShare/Teacher/liulei/Dataset//vid'
        self.imagenet_lmdb_dir = '/mnt/dataShare/Teacher/liulei/Dataset//vid_lmdb'
        self.lasher_train_dir = '/mnt/dataShare/Teacher/liulei/Dataset//LasHeR'
        self.lasher_test_dir = '/mnt/dataShare/Teacher/liulei/Dataset//LasHeR'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.UAV_RGBT_dir = '/mnt/dataShare/Teacher/liulei/Dataset//VTUAV'
