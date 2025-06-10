import os
import numpy as np
import h5py
import pandas as pd

def task_count_matching_num(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    df = pd.read_csv(os.path.join(work_dir, params["input"]["image_pair"]))
    updated_dict = {
        "key1": [],
        "key2": [],
        "sim": [],
        "dir1": [],
        "dir2": [],
        "match_num": []
    }

    if "keypoints" in params["input"]:
        keypoints = {}
        with h5py.File(os.path.join(work_dir, params["input"]["keypoints"]), mode="r") as f_keypoints:
            for key in f_keypoints.keys():
                keypoints[key] = f_keypoints[key][...]

        updated_dict['pct_1'] = []
        updated_dict['pct_2'] = []

    with h5py.File(os.path.join(work_dir, params["input"]["matches"]), mode="r") as f_matches:
        for i, row in df.iterrows():
            key1 = row["key1"]
            key2 = row["key2"]
            if key1 not in f_matches or key2 not in f_matches[key1]:
                continue
            match_num = f_matches[key1][key2].shape[0]
            
            updated_dict["key1"].append(key1)
            updated_dict["key2"].append(key2)
            updated_dict["sim"].append(row["sim"])
            updated_dict["dir1"].append(row["dir1"])
            updated_dict["dir2"].append(row["dir2"])
            updated_dict["match_num"].append(match_num)
            if "keypoints" in params["input"]:
                updated_dict["pct_1"].append(match_num / len(keypoints[key1]))
                updated_dict["pct_2"].append(match_num / len(keypoints[key2]))
    
    dst_df = pd.DataFrame.from_dict(updated_dict)
    output_path = os.path.join(work_dir, params["output"])
    dst_df.to_csv(output_path, index=False)
