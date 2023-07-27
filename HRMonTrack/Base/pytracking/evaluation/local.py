from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/data/zzy/ablation/V11_V909_swin_base_rank/ltr/checkpoints/ltr/transt/transt/TransT_ep0081.pth.tar'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/data/zhu_19/TransT-test/17-09-22/pytracking/result_plots/'
    settings.results_path = '/data/zzy/ablation/V11_V909_swin_base_rank/pytracking/tracking_results_0081_new/'    # Where to store tracking results
    # settings.segmentation_path = '/data/zhu_19/TransT-test/17-09-22/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

