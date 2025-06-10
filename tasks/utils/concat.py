import os
import numpy as np
import h5py

def task_concat(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]

    # concat keypoints
    keypoints = {}
    with h5py.File(os.path.join(work_dir, params["input"]["keypoints"][0]), mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            keypoints[key] = f_keypoints[key][...]
    
    offsets = []
    for i, p in enumerate(params["input"]["keypoints"][1:]):
        offsets.append({})
        with h5py.File(os.path.join(work_dir, p), mode="r") as f_keypoints:
            for key in f_keypoints.keys():
                if key in keypoints:
                    offsets[i][key] = keypoints[key].shape[0]
                    keypoints[key] = np.concatenate([keypoints[key], f_keypoints[key][...]])
                else:
                    offsets[i][key] = 0
                    keypoints[key] = f_keypoints[key][...]

    dst_keypoints_h5_path = os.path.join(work_dir, params["output"]["keypoints"])
    with h5py.File(dst_keypoints_h5_path, mode="w") as f_keypoints:
        for k, v in keypoints.items():
            f_keypoints[k] = v
    del keypoints
    

    # concat matches
    matches = {}
    with h5py.File(os.path.join(work_dir, params["input"]["matches"][0]), mode="r") as f_matches:
        for key1 in f_matches.keys():
            matches[key1] = {}
            for key2 in f_matches[key1].keys():
                matches[key1][key2] = f_matches[key1][key2][...]
    
    for i, p in enumerate(params["input"]["matches"][1:]):
        with h5py.File(os.path.join(work_dir, p), mode="r") as f_matches:
            for key1 in f_matches.keys():
                if key1 not in matches:
                    matches[key1] = {}
                
                for key2 in f_matches[key1].keys():
                    m = f_matches[key1][key2][...]
                    m[:, 0] += offsets[i][key1]
                    m[:, 1] += offsets[i][key2]

                    if key2 in matches[key1]:
                        matches[key1][key2] = np.concatenate([matches[key1][key2], m])
                    else:
                        matches[key1][key2] = m
    
    matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
    with h5py.File(matches_h5_path, mode="w") as f_matches:
        for key1 in matches.keys():
            for key2 in matches[key1].keys():
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches[key1][key2])