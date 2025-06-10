import os
import numpy as np
from pathlib import Path
import pandas as pd
import h5py
from tasks.matching.core.computing_keypoints import *
from tasks.matching.core.match import *
import shutil

def rem_confirmed_image_pairs(image_pairs, confirmed_pairs):
    updated_image_pairs = []
    for pair in image_pairs:
        key1 = " ".join(pair[0].split(" ")[:-1])
        key2 = " ".join(pair[1].split(" ")[:-1])
        if (key1, key2) in confirmed_pairs:
            continue
        updated_image_pairs.append(pair)
    return updated_image_pairs

def task_rotate_matching_find_best(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    temp_dir = os.path.join(work_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    target_directions = [0]
    if params["enable_rotation"]:
        ref_directions = [0, 90, 180, 270]
    else:
        ref_directions = [0]
    sufficient_matching_num = params["sufficient_matching_num"] if "sufficient_matching_num" in params else np.inf

    image_pair_df = pd.read_csv(os.path.join(work_dir, params["input"]["image_pair"]))

    matcher_type = params["matcher"]
    if matcher_type == "LightGlue":
        images_dir = params["data_dict"][0].parent
        img1_list = image_pair_df["key1"].values.tolist()
        img2_list = image_pair_df["key2"].values.tolist()
        
        # create path list of all combinations
        all_image_paths = []
        for dir1 in target_directions:
            dir1_list = [dir1]*len(img1_list)
            for dir2 in ref_directions:
                dir2_list = [dir2]*len(img2_list)
                image_paths = list(set(list(zip(img1_list, dir1_list)) + list(zip(img2_list, dir2_list))))
                image_paths = [(Path(os.path.join(str(images_dir), p[0])), p[1]) for p in image_paths]
                all_image_paths += image_paths

        all_image_paths = list(set(all_image_paths))
        
        # Detect keypoints
        extractor_type = params["extractor"]
        if extractor_type in ["aliked", "disk", "sift", "superpoint"]:
            keypoints, descriptors = detect_keypoints(
                all_image_paths,
                extractor_type,
                **params["keypoint_detection_args"],
                device=params["device"],
            )
        # elif extractor_type == "superpoint":
        #     keypoints, descriptors = detect_keypoints_superpoint(
        #         image_paths,
        #         **params["keypoint_detection_args"],
        #         device=params["device"],
        #     )
        else:
            raise NotImplemented
        
        # Save to h5 (keypoints & descriptions)
        keypoints_h5_path = os.path.join(temp_dir, params["output"]["keypoints"])
        descriptions_h5_path = os.path.join(temp_dir, params["output"]["descriptions"])
        with h5py.File(keypoints_h5_path, mode="w") as f_keypoints:
            for k, v in keypoints.items():
                f_keypoints[k] = v

        with h5py.File(descriptions_h5_path, mode="w") as f_descriptors:
            for k, v in descriptors.items():
                f_descriptors[k] = v
        

        # Matching all combinations
        max_matching_num = {}
        best_dirs = {}
        for key1 in img1_list:
            max_matching_num[key1] = {}
            best_dirs[key1] = {}
            for key2 in img2_list:
                max_matching_num[key1][key2] = 0
                best_dirs[key1][key2] = {
                    "dir1": 0,
                    "dir2": 0
                }

        confirmed_pairs = []
        for dir1 in target_directions:
            dir1_list = [dir1]*len(img1_list)
            for dir2 in ref_directions:
                dir2_list = [dir2]*len(img2_list)
                print(f"Matching {dir1} - {dir2}")

                # Matching keypoints
                dir_img1_list = [p1+" "+str(p2) for p1, p2 in zip(img1_list, dir1_list)]
                dir_img2_list = [p1+" "+str(p2) for p1, p2 in zip(img2_list, dir2_list)]
                image_pairs = list(zip(dir_img1_list, dir_img2_list))
                image_pairs = rem_confirmed_image_pairs(image_pairs, confirmed_pairs)
                print(f"pair_num -> {len(image_pairs)}")

                matches = keypoint_matcing_LG(
                    image_pairs,
                    keypoints_h5_path,
                    descriptions_h5_path,
                    extractor_type,
                    **params["keypoint_matching_args"],
                    device=params["device"]
                )

                # Save to h5 (matches)
                matches_h5_path = os.path.join(temp_dir, params["output"]["matches"]+f"_{dir1}-{dir2}")
                with h5py.File(matches_h5_path, mode="w") as f_matches:
                    for key1 in matches.keys():
                        for key2 in matches[key1].keys():
                            group  = f_matches.require_group(key1)
                            group.create_dataset(key2, data=matches[key1][key2])

                            match_num = matches[key1][key2].shape[0]
                            _key1 = " ".join(key1.split(" ")[:-1])
                            _key2 = " ".join(key2.split(" ")[:-1])
                            if match_num > max_matching_num[_key1][_key2]:
                                max_matching_num[_key1][_key2] = match_num
                                best_dirs[_key1][_key2]["dir1"] = dir1
                                best_dirs[_key1][_key2]["dir2"] = dir2

                                if match_num > sufficient_matching_num:
                                    confirmed_pairs.append((_key1, _key2))

        # Update best_dir
        min_matches = params["keypoint_matching_args"]["min_matches"]
        best_image_pair_dict = {
            "key1": [],
            "key2": [],
            "sim": [],
            "dir1": [],
            "dir2": [],
            "match_num": [],
        }
        for i, row in image_pair_df.iterrows():
            key1 = row["key1"]
            key2 = row["key2"]
            sim = row["sim"]
            if max_matching_num[key1][key2] < min_matches:
                continue

            best_image_pair_dict["key1"].append(key1)
            best_image_pair_dict["key2"].append(key2)
            best_image_pair_dict["sim"].append(sim)
            best_image_pair_dict["dir1"].append(best_dirs[key1][key2]["dir1"])
            best_image_pair_dict["dir2"].append(best_dirs[key1][key2]["dir2"])
            best_image_pair_dict["match_num"].append(max_matching_num[key1][key2])
        image_pair_df = pd.DataFrame.from_dict(best_image_pair_dict)
        image_pair_df.to_csv(os.path.join(work_dir, params["output"]["image_pair_csv"]), index=False)

        # Reset image path
        img1_list = image_pair_df["key1"].values.tolist()
        img2_list = image_pair_df["key2"].values.tolist()
        dir1_list = image_pair_df["dir1"].values.tolist()
        dir2_list = image_pair_df["dir2"].values.tolist()
        image_paths = list(set(list(zip(img1_list, dir1_list)) + list(zip(img2_list, dir2_list))))
        image_paths = [(Path(os.path.join(str(images_dir), p[0])), p[1]) for p in image_paths]

        # PostProcess(keypoints)
        offsets = {}
        keypoints = {}
        with h5py.File(keypoints_h5_path, mode="r") as f_keypoints:
            for data in image_paths:
                path = data[0]
                dir = data[1]
                key = path.name

                kpt = f_keypoints[path.name+" "+str(dir)][...]
                if dir == 0:
                    rotated_kpt = kpt
                elif dir == 90:
                    img = cv2.imread(str(images_dir / path.name))
                    w, h, _ = img.shape    # reverse h, w, since img is rotated 90 degree
                    rotated_kpt = np.zeros_like(kpt)
                    rotated_kpt[:, 0] = kpt[:, 1]
                    rotated_kpt[:, 1] = w - 1 - kpt[:, 0]
                elif dir == 180:
                    img = cv2.imread(str(images_dir / path.name))
                    h, w, _ = img.shape
                    rotated_kpt = np.zeros_like(kpt)
                    rotated_kpt[:, 0] = w - 1 - kpt[:, 0]
                    rotated_kpt[:, 1] = h - 1 - kpt[:, 1]
                elif dir == 270:
                    img = cv2.imread(str(images_dir / path.name))
                    w, h, _ = img.shape    # reverse h, w, since img is rotated 90 degree
                    rotated_kpt = np.zeros_like(kpt)
                    rotated_kpt[:, 0] = h - 1 - kpt[:, 1]
                    rotated_kpt[:, 1] = kpt[:, 0]
                
                if key in keypoints:
                    offsets[key][dir] = keypoints[key].shape[0]
                    keypoints[key] = np.concatenate([keypoints[key], rotated_kpt])
                else:
                    offsets[key] = {}
                    offsets[key][dir] = 0
                    keypoints[key] = rotated_kpt
        
        keypoints_h5_path = os.path.join(work_dir, params["output"]["keypoints"])
        with h5py.File(keypoints_h5_path, mode="w") as f_keypoints:
            for k, v in keypoints.items():
                f_keypoints[k] = v
        
        # PostProcess(matches)
        matches = {}
        for target_dir in target_directions:
            for ref_dir in ref_directions:
                matches_h5_path = os.path.join(temp_dir, params["output"]["matches"]+f"_{target_dir}-{ref_dir}")
                with h5py.File(matches_h5_path, mode="r") as f_matches:
                    for i, row in image_pair_df.iterrows():
                        key1 = row["key1"]
                        key2 = row["key2"]
                        dir1 = int(row["dir1"])
                        dir2 = int(row["dir2"])
                        if dir1!=target_dir or dir2!=ref_dir:
                            continue

                        dir_key1 = key1 + " " + str(dir1)
                        dir_key2 = key2 + " " + str(dir2)
                        if dir_key1 in f_matches and dir_key2 in f_matches[dir_key1]:
                            match = f_matches[dir_key1][dir_key2][...]
                            offset1 = offsets[key1][int(dir1)]
                            offset2 = offsets[key2][int(dir2)]
                            match[:, 0] += offset1
                            match[:, 1] += offset2

                            if key1 not in matches:
                                matches[key1] = {}
                            matches[key1][key2] = match

        
        matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])
        
        # Remove tmp dir
        shutil.rmtree(temp_dir)


    else:
        raise NotImplemented