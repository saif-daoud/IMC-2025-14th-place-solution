import os
import json
import h5py

def task_extract_inliner_matching_points(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]

    json_path = os.path.join(work_dir, params["input"]["rects"])
    with open(json_path, "r") as f:
        rects = json.load(f)
    
    keypoints = {}
    with h5py.File(os.path.join(work_dir, params["input"]["keypoints"]), mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            keypoints[key] = f_keypoints[key][...]
    
    matches = {}
    with h5py.File(os.path.join(work_dir, params["input"]["matches"]), mode="r") as f_matches:
        for key1 in f_matches.keys():
            matches[key1] = {}
            for key2 in f_matches[key1].keys():
                m = f_matches[key1][key2][...]
                keypoints1 = keypoints[key1]
                keypoints2 = keypoints[key2]
                rect1 = rects[key1]
                rect2 = rects[key2]

                mask1 = ((keypoints1[:, 0] > rect1[0]) 
                         & (keypoints1[:, 1] > rect1[1])
                         & (keypoints1[:, 0] < rect1[2])
                         & (keypoints1[:, 1] < rect1[3]))
                mask2 = ((keypoints2[:, 0] > rect2[0]) 
                         & (keypoints2[:, 1] > rect2[1])
                         & (keypoints2[:, 0] < rect2[2])
                         & (keypoints2[:, 1] < rect2[3]))
                match_mask = mask1[m[:, 0]] & mask2[m[:, 1]]
                inliner_match = m[match_mask]
                matches[key1][key2] = inliner_match
    
    matches_h5_path = os.path.join(work_dir, params["output"])
    with h5py.File(matches_h5_path, mode="w") as f_matches:
        for key1 in matches.keys():
            for key2 in matches[key1].keys():
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches[key1][key2])