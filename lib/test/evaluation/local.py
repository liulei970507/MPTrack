from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    root_dir = '/data/Share/Teacher/liulei/Code/MPTrack'
    output_dir = root_dir + '/output'

    # Set your local paths here.

    settings.check_dir = output_dir
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/dataShare/Teacher/liulei/Dataset/got10k_lmdb'
    settings.got10k_path = '/mnt/dataShare/Teacher/liulei/Dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/mnt/dataShare/Teacher/liulei/Dataset/itb'
    settings.lasot_extension_subset_path_path = '/mnt/dataShare/Teacher/liulei/Dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = '/mnt/dataShare/Teacher/liulei/Dataset/lasot_lmdb'
    settings.lasot_path = '/mnt/dataShare/Teacher/liulei/Dataset/lasot'
    settings.network_path = output_dir + '/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/dataShare/Teacher/liulei/Dataset/nfs'
    settings.otb_path = '/mnt/dataShare/Teacher/liulei/Dataset/otb'
    settings.prj_dir = root_dir
    settings.result_plot_path = output_dir + '/test/result_plots'
    settings.results_path = output_dir + '/test/tracking_results'    # Where to store tracking results
    settings.save_dir = output_dir
    settings.segmentation_path = output_dir + '/test/segmentation_results'
    settings.tc128_path = '/mnt/dataShare/Teacher/liulei/Dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/dataShare/Teacher/liulei/Dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/dataShare/Teacher/liulei/Dataset/trackingnet'
    settings.uav_path = '/mnt/dataShare/Teacher/liulei/Dataset/uav'
    settings.vot18_path = '/mnt/dataShare/Teacher/liulei/Dataset/vot2018'
    settings.vot22_path = '/mnt/dataShare/Teacher/liulei/Dataset/vot2022'
    settings.vot_path = '/mnt/dataShare/Teacher/liulei/Dataset/VOT2019'
    settings.youtubevos_dir = ''
    settings.gtot_path = "/mnt/dataShare/Teacher/liulei/Dataset/GTOT/"
    settings.rgbt210_path = "/mnt/dataShare/Teacher/liulei/Dataset/RGBT210/"
    settings.rgbt234_path = "/mnt/dataShare/Teacher/liulei/Dataset/RGBT234/"
    settings.lasher_path = "/mnt/dataShare/Teacher/liulei/Dataset/LasHeR/"
    settings.vtuav_path = "/mnt/dataShare/Teacher/liulei/Dataset/VTUAV/"

    return settings
