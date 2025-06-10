import os
import numpy as np
import cv2
import h5py
import pickle

def task_ransac(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    ransac_params = {
        "param1": 0.2,
        "param2": 0.999,
        "maxIters": 5000
    }
    ransac_params.update(params["ransac_params"])

    keypoints = {}
    with h5py.File(os.path.join(work_dir, params["input"]["keypoints"]), mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            keypoints[key] = f_keypoints[key][...]
    
    # RANSAC
    matches = {}
    fms = {}
    with h5py.File(os.path.join(work_dir, params["input"]["matches"]), mode="r") as f_matches:
        for key1 in f_matches.keys():
            for key2 in f_matches[key1].keys():
                m = f_matches[key1][key2][...]
                keypoints1 = keypoints[key1][m[:, 0]]
                keypoints2 = keypoints[key2][m[:, 1]]
                try:
                    F, inliers = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.USAC_MAGSAC, ransac_params["param1"], ransac_params["param2"], ransac_params["maxIters"])
                except:
                    continue
                fms[(key1, key2)] = F

                inliers = inliers > 0
                sampled_matches = m[inliers[:,0]]

                if sampled_matches.shape[0] < params["min_matches"]:
                    continue

                if key1 not in matches:
                    matches[key1] = {}
                matches[key1][key2] = sampled_matches
    
    # Save h5
    matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
    with h5py.File(matches_h5_path, mode="w") as f_matches:
        for key1 in matches.keys():
            for key2 in matches[key1].keys():
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches[key1][key2])
    
    # Save fms
    with open(os.path.join(work_dir, params["output"]["fms"]), "wb") as f:
        pickle.dump(fms, f)