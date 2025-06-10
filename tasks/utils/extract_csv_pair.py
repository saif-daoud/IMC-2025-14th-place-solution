import os
import numpy as np
import h5py
import pandas as pd

def task_extract_csv_pair(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    df = pd.read_csv(os.path.join(work_dir, params["input"]["image_pair"]))
    image_pairs = []
    for i, row in df.iterrows():
        key1 = row["key1"]
        key2 = row["key2"]
        image_pairs.append((key1, key2))
        image_pairs.append((key2, key1))

    matches = {}
    with h5py.File(os.path.join(work_dir, params["input"]["matches"]), mode="r") as f_matches:
        for key1 in f_matches.keys():
            for key2 in f_matches[key1].keys():
                if (key1, key2) in image_pairs:
                    if key1 not in matches:
                        matches[key1] = {}
                    matches[key1][key2] = f_matches[key1][key2][...]
    

    matches_h5_path = os.path.join(work_dir, params["output"])
    with h5py.File(matches_h5_path, mode="w") as f_matches:
        for key1 in matches.keys():
            for key2 in matches[key1].keys():
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches[key1][key2])