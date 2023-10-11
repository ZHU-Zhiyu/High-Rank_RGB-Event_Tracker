class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/pretrained_networks'
        self.coesot_dir = '/data/zhu_19/COESOT_Remote'
        self.coesot_val_dir = '/data/zhu_19/COESOT_Remote'
        self.fe108_dir = '/data/zhu_19/FE108/train'
        self.fe108_val_dir = '/data/zhu_19/FE108/test'
        self.visevent_dir = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/data/VisEvent/train'
        self.visevent_val_dir = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/data/VisEvent/test'
