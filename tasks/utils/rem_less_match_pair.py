import os
import numpy as np
import h5py

def task_rem_less_match_pair(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    th_matching_num = params["th_matching_num"]
    matches = {}
    all_pair_num = 0
    removed_pair_num = 0
    with h5py.File(os.path.join(work_dir, params["input"]), mode="r") as f_matches:
        for key1 in f_matches.keys():
            matches[key1] = {}
            for key2 in f_matches[key1].keys():
                all_pair_num += 1
                if f_matches[key1][key2][...].shape[0] < th_matching_num:
                    continue

                if key1 not in matches:
                    matches[key1] = {}
                matches[key1][key2] = f_matches[key1][key2][...]
                removed_pair_num += 1
    
    print(f"pair_num {all_pair_num} -> {removed_pair_num}")
    matches_h5_path = os.path.join(work_dir, params["output"])
    with h5py.File(matches_h5_path, mode="w") as f_matches:
        for key1 in matches.keys():
            for key2 in matches[key1].keys():
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches[key1][key2])