import os
import time
import pandas as pd
import numpy as np
import gc
import kornia as K

from tasks.get_pair.get_image_pair_exhaustive import task_get_image_pair_exhaustive
from tasks.get_pair.get_image_pair_exhaustive_within_component import task_get_image_pair_exhaustive_within_component
from tasks.get_pair.get_image_pair_DINO import task_get_image_pair_DINO
from tasks.get_pair.get_image_pair_kNN import task_get_image_pair_kNN
from tasks.get_pair.get_transparent_pair import task_get_transparent_pair
from tasks.matching.matching import task_matching
from tasks.matching.rotate_matching_find_best import task_rotate_matching_find_best
from tasks.crop.sfm_mkpc import task_sfm_mkpc
from tasks.crop.pair_mkpc import task_pair_mkpc
from tasks.crop.transparent_crop import task_transparent_crop
from tasks.utils.ransac import task_ransac
from tasks.utils.concat import task_concat
from tasks.utils.rem_less_match_pair import task_rem_less_match_pair
from tasks.utils.count_matching_num import task_count_matching_num
from tasks.utils.extract_inliner_matching_points import task_extract_inliner_matching_points
from tasks.utils.extract_csv_pair import task_extract_csv_pair
# from tasks.utils.estimate_rot import task_estimate_rot
from tasks.utils.get_exif import task_get_exif
from tasks.utils.get_dino_embeddings import task_get_dino_embeddings
from tasks.utils.copy_dino_embeddings import task_copy_dino_embeddings
from tasks.utils.threading import single_threaded
from tasks.utils.rem_less_match_pair_pct import task_rem_less_match_pair_pct

task_map = {
    "get_image_pair_exhaustive": task_get_image_pair_exhaustive,
    "get_image_pair_exhaustive_within_component": task_get_image_pair_exhaustive_within_component,
    "get_image_pair_DINO": task_get_image_pair_DINO,
    "get_image_pair_kNN": task_get_image_pair_kNN,
    "get_transparent_pair": task_get_transparent_pair,

    "matching": task_matching,
    "rotate_matching_find_best": task_rotate_matching_find_best,

    "sfm_mkpc": task_sfm_mkpc,
    "pair_mkpc": task_pair_mkpc,
    "transparent_crop": task_transparent_crop,
    
    "ransac": task_ransac,
    "concat": task_concat,
    "rem_less_match_pair": task_rem_less_match_pair,
    "rem_less_match_pair_pct": task_rem_less_match_pair_pct,
    "count_matching_num": task_count_matching_num,
    "extract_inliner_matching_points": task_extract_inliner_matching_points,
    "extract_csv_pair": task_extract_csv_pair,
    # "estimate_rot": task_estimate_rot,
    "get_exif": task_get_exif,
    "get_dino_embeddings": task_get_dino_embeddings,
    "copy_dino_embeddings": task_copy_dino_embeddings,
}

class Pipeline():
    def __init__(self, data_dict, work_dir, input_dir_root, pipeline_config, device_id):
        self.device = K.utils.get_cuda_device_if_available(device_id)
        print(f"device: {self.device}")
        self.data_dict = data_dict
        self.work_dir = work_dir
        self.input_dir_root = input_dir_root
        self.pipeline_config = pipeline_config
        self.processing_times = {
            "task": [],
            "comment": [],
            "processing_time": []
        }

    def exec(self):
        all_processing_time = 0
        for p in self.pipeline_config:
            task = p["task"]
            comment = p["comment"]
            p["params"]["device"] = self.device
            p["params"]["data_dict"] = self.data_dict
            p["params"]["work_dir"] = self.work_dir
            p["params"]["input_dir_root"] = self.input_dir_root
            if "pdb" in p and p["pdb"]:
                p["params"]["pdb"] = True
            else:
                p["params"]["pdb"] = False
            
            start = time.time()
            print(f"===== [{task}] {comment} =====")
            task_map[task](p["params"])
            gc.collect()
            print("====================")
            end = time.time()

            self.processing_times["task"].append(task)
            self.processing_times["comment"].append(comment)
            self.processing_times["processing_time"].append(end-start)
            all_processing_time += end-start
        
        self.processing_times["task"].append("All")
        self.processing_times["comment"].append("")
        self.processing_times["processing_time"].append(all_processing_time)
        processing_times_df = pd.DataFrame.from_dict(self.processing_times)
        processing_times_df.to_csv(os.path.join(self.work_dir, "processing_time.csv"), index=False)