import multiprocessing
import argparse
import os
import os.path as osp
from glob import glob
from time import time
import pandas as pd
import shutil
import json
import gc
import torch
from pathlib import Path
from config import Config
from pipeline import Pipeline
from reconstruction import reconstruction

import warnings
warnings.filterwarnings("ignore")

def parse_sample_submission(
    base_path: str,
    input_csv: str,
    target_datasets: list,
) -> dict[dict[str, list[Path]]]:
    """Construct a dict describing the test data as 
    
    {"dataset": {"scene": [<image paths>]}}
    """
    data_dict = {}
    if 'train' in osp.basename(input_csv):
        with open(input_csv, "r") as f:
            for i, l in enumerate(f):
                # Skip header
                if i == 0:
                    print("header:", l)

                if l and i > 0:
                    dataset, scene, image_path, _, _ = l.strip().split(',')
                    if target_datasets is not None and dataset not in target_datasets:
                        continue

                    if dataset not in data_dict:
                        data_dict[dataset] = {}
                    if scene not in data_dict[dataset]:
                        data_dict[dataset][scene] = []
                    data_dict[dataset][scene].append(Path(os.path.join(base_path, dataset, image_path)))

        for dataset in data_dict:
            for scene in data_dict[dataset]:
                print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
        data_dict = {dataset: [im_path for im_paths in scenes.values() for im_path in im_paths] for dataset, scenes in data_dict.items()}
    else:
        datasets = os.listdir(base_path)
        for dataset in datasets:
            if target_datasets is not None and dataset not in target_datasets:
                continue
            data_dict[dataset] = []
            for p in glob(osp.join(base_path, dataset, "*.png")):
                data_dict[dataset].append(Path(p))
            print(f"{dataset} -> {len(data_dict[dataset])} images")

    return data_dict

def worker_reconstruction(input_queue, submission_path_list):
    while True:
        reconstruction_inputs = input_queue.get()
        if reconstruction_inputs is None:
            break
        data_dict, dataset, work_dir, colmap_mapper_options, thr_config = reconstruction_inputs
        submission_path = reconstruction(data_dict, dataset, work_dir, colmap_mapper_options, thr_config)
        submission_path_list.append(submission_path)

def run(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--target_datasets", nargs="*", required=False)
    parser.add_argument("--output_dir", required=False)
    args = parser.parse_args()

    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.target_datasets:
        config.target_datasets = args.target_datasets

    # Check output_dir
    if config.check_exist_dir:
        if os.path.isdir(config.output_dir):
            raise Exception(f"{config.output_dir} is already exists.")

    os.makedirs(config.output_dir, exist_ok=True)
    base_path = os.path.join(config.input_dir_root, "train" if "train" in osp.basename(config.input_csv) else "test")
    feature_dir = os.path.join(config.output_dir, "feature_outputs")
    shutil.copy(config.pipeline_json, config.output_dir)

    data_dict = parse_sample_submission(base_path, config.input_csv, config.target_datasets)
    datasets = list(data_dict.keys())
    
    manager = multiprocessing.Manager()
    submission_path_list = manager.list()

    input_queue = multiprocessing.Queue()
    worker_reconstruction_process = multiprocessing.Process(target=worker_reconstruction, args=(input_queue, submission_path_list))
    worker_reconstruction_process.start()
    for dataset in datasets:
        work_dir = Path(os.path.join(feature_dir, dataset))
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Exec Pipeline
        with open(config.pipeline_json, "r") as f:
            pipeline_config = json.load(f)
        pipeline = Pipeline(data_dict[dataset], work_dir, config.input_dir_root, pipeline_config, args.device_id)
        pipeline.exec()

        # Reconstruction & Save CSV
        print("Start Reconstruction")            
        torch.cuda.empty_cache()
        gc.collect()
        input_queue.put((data_dict, dataset, work_dir, config.colmap_mapper_options, config.thr_config))
    input_queue.put(None)
    worker_reconstruction_process.join()
    # Concat Submission
    if len(submission_path_list) > 0:
        submission_df_list = [pd.read_csv(p) for p in submission_path_list]
        submission_df = pd.concat(submission_df_list).reset_index(drop=True)
    else:
        submission_df = pd.DataFrame.from_dict({
            "image_id": [],
            "dataset": [],
            "scene": [],
            "image": [],
            "rotation_matrix": [],
            "translation_vector": []
        })
    submission_df.to_csv(os.path.join(config.output_dir, "submission.csv"), index=False)

    # # Compute Metric
    # if "train" in config.input_csv:
    #     from metric import score
    #     t = time()
    #     final_score, dataset_scores = score(
    #         gt_csv=config.input_csv,
    #         # user_csv=r"E:\competitions\kaggle\Image Matching Challenge 2025\data\output\exp1\debug\feature_outputs\imc2023_haiper_cluster0\submission.csv",
    #         user_csv=osp.join(config.output_dir, "submission.csv"),
    #         thresholds_csv=osp.join(config.input_dir_root, 'train_thresholds.csv'),
    #         mask_csv=None,
    #         target_datasets=config.target_datasets,
    #         inl_cf=0,
    #         strict_cf=-1,
    #         verbose=True,
    #     )
    #     print(f'Computed metric in: {time() - t:.02f} sec.')

if __name__ == '__main__':
    cfg = Config
    run(cfg)