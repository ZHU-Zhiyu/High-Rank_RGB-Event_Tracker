from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/data/zzy/ablation/V9_Swin_mon_02_accu_box_09/ltr/checkpoints/ltr/transt/transt/TransT_ep0063.pth.tar'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/data/zhu_19/TransT-test/17-09-22/pytracking/result_plots/'
    settings.results_path = '/data/zzy/ablation/V9_Swin_mon_02_accu_box_09/pytracking/tracking_results_0063_new/'    # Where to store tracking results
    # settings.segmentation_path = '/data/zhu_19/TransT-test/17-09-22/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

