from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.local import local_env_settings
from lib.config.ceutrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ceutrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)
    local_set = local_env_settings()

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # /data/zhu_19/COESOT-Large-COESOT/CEUTrack/output/checkpoints/train/ceutrack/ceutrack_coesot/
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/ceutrack/ceutrack_coesot/CEUTrack_ep%04d.pth.tar" %
                                     (int(yaml_name.split('_')[-1])))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
