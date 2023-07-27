from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.coesot_path = '/data1/zhu_19/COESOT'
    # settings.prj_dir = '/data/zzy/OS_Track_study/COESOT-main-Out/'
    settings.fe108_path = '/data/zzy/OS_Track_study/COESOT-main-Out/CEUTrack/data/FE108'
    settings.network_path = '/data/zzy/OS_Track_study/COESOT-main-Out/CEUTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/data/zzy/OS_Track_study/COESOT-mask-rank-21/CEUTrack'
    settings.result_plot_path = settings.prj_dir + '/output/test/result_plots'
    settings.test_epoch = 50
    settings.results_path = settings.prj_dir + '/output/test/tracking_results_{}'.format(settings.test_epoch)    # Where to store tracking results
    settings.save_dir = settings.prj_dir + '/output'
    settings.segmentation_path = '/data/zzy/OS_Track_study/COESOT-main-Out/CEUTrack/output/test/segmentation_results'
    settings.visevent_path = '/data/zzy/OS_Track_study/COESOT-main-Out/CEUTrack/data/VisEvent'

    return settings

