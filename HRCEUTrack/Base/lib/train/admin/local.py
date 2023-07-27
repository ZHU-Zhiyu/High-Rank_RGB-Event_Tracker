class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/zzy/OS_Track_study/COESOT-mask-rank-21/CEUTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + 'tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/zzy/pretrained_models'
        self.coesot_dir = '/data0/zhu_19/COETracking/train_subset'
        self.coesot_val_dir = '/data0/zhu_19/COETracking/train_subset'
        self.fe108_dir = self.workspace_dir + 'CEUTrack/data/FE108/train'
        self.fe108_val_dir = self.workspace_dir + 'CEUTrack/data/FE108/test'
        self.visevent_dir = self.workspace_dir + 'CEUTrack/data/VisEvent/train'
        self.visevent_val_dir = self.workspace_dir + 'CEUTrack/data/VisEvent/test'
