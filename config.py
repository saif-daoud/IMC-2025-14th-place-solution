import os
import os.path as osp

class Config:
    input_dir_root = osp.join("data", "input")
    output_dir = osp.join("data", "output", "exp9", "debug")
    check_exist_dir = False
    
    input_csv = osp.join(input_dir_root, "train_labels.csv")

    thr_config = {
        'thr': 0.6,
        'tolerance': 0.5,
        'clustering_threshold': 0.85,
        'use_centroids': False,
        'tries_for_second_img': 3,
        'rem_only_registered_imgs_having_all_pairs_registered': True,
        'fix_pair_for_iter': True, 
        'fix_pair_for_cluster': True,
        'verbose': False
    }

    use_gt_clusters = False

    # target_datasets = None
    target_datasets = ['imc2023_haiper', 'ETs']
    # target_datasets = ['imc2023_haiper', 'imc2023_heritage', 'ETs']
    # target_datasets = ['imc2023_heritage', 'imc2023_haiper', 'amy_gardens', 'imc2023_theather_imc2024_church', 'imc2024_dioscuri_baalshamin', 'stairs', 'ETs']
    # target_datasets += ['pt_piazzasanmarco_grandplace', 'pt_sacrecoeur_trevi_tajmahal', 'pt_stpeters_stpauls', 'pt_brandenburg_british_buckingham']
    # target_datasets = ['amy_gardens', 'ETs', 'fbk_vineyard', 'imc2023_haiper', 'imc2023_heritage', 'imc2023_theather_imc2024_church', 
    #                    'imc2024_dioscuri_baalshamin', 'imc2024_lizard_pond', 'pt_brandenburg_british_buckingham', 
    #                    'pt_piazzasanmarco_grandplace', 'pt_sacrecoeur_trevi_tajmahal', 'pt_stpeters_stpauls', 'stairs']

    pipeline_json = osp.join("exp", "exp9", "pipeline.json")
    
    colmap_mapper_options = {
        "min_model_size": 3, # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 2,
        "num_threads": 1
    }
