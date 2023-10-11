from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.test_epoch = 45
    settings.coesot_path = '/data/zhu_19/COESOT_Remote_test'
    settings.fe108_path = '/data/zhu_19/FE108/test'
    settings.network_path = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack'
    settings.result_plot_path = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/output/test/result_plots'
    settings.results_path = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/output'
    settings.segmentation_path = '/data/zhu_19/COESOT-Large-COESOT/CEUTrack/output/test/segmentation_results'
    settings.visevent_path = '/data/zhu_19/FE108/train/VisEvent'

    return settings

