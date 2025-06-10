import os
from pathlib import Path
import pandas as pd
import h5py
from tasks.matching.core.computing_keypoints import *
from tasks.matching.core.match import *
from tasks.matching.core.match_lg_pair import *
import shutil
import json

def task_matching(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    image_pair_df = pd.read_csv(os.path.join(work_dir, params["input"]["image_pair"]))

    if "rects" in params["input"]:
        with open(os.path.join(work_dir, params["input"]["rects"])) as f:
            rects = json.load(f)
    else:
        rects = None

    matcher_type = params["matcher"]
    if matcher_type == "LightGlue":
        temp_dir = os.path.join(work_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        images_dir = params["data_dict"][0].parent
        img1_list = image_pair_df["key1"].values.tolist()
        img2_list = image_pair_df["key2"].values.tolist()
        dir1_list = image_pair_df["dir1"].values.tolist()
        dir2_list = image_pair_df["dir2"].values.tolist()
        image_paths = list(set(list(zip(img1_list, dir1_list)) + list(zip(img2_list, dir2_list))))
        image_paths = [(Path(os.path.join(str(images_dir), p[0])), p[1]) for p in image_paths]
        
        # Detect keypoints
        extractor_type = params["extractor"]
        if extractor_type in ["aliked", "disk", "sift", "superpoint"]:
            keypoints, descriptors = detect_keypoints(
                image_paths,
                extractor_type,
                **params["keypoint_detection_args"],
                rects = rects,
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
        
        # Matching keypoints
        dir_img1_list = [p1+" "+str(p2) for p1, p2 in zip(img1_list, dir1_list)]
        dir_img2_list = [p1+" "+str(p2) for p1, p2 in zip(img2_list, dir2_list)]
        image_pairs = list(zip(dir_img1_list, dir_img2_list))
        matches = keypoint_matcing_LG(
            image_pairs,
            keypoints_h5_path,
            descriptions_h5_path,
            extractor_type,
            **params["keypoint_matching_args"],
            device=params["device"]
        )

        # Save to h5 (matches)
        matches_h5_path = os.path.join(temp_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])
        
        # PostProcess(keypoints)
        offsets = {}
        keypoints = {}
        with h5py.File(keypoints_h5_path, mode="r") as f_keypoints:
            for data in image_paths:
                path = data[0]
                dir = data[1]
                key = path.name

                img = cv2.imread(str(images_dir / path.name))
                img_shape = img.shape
                if rects is not None and path.name in rects:
                    rect = rects[path.name]
                else:
                    rect = [0, 0, img_shape[1], img_shape[0]]
                height = rect[3]-rect[1]
                width = rect[2]-rect[0]

                kpt = f_keypoints[path.name+" "+str(dir)][...]
                # Revert Rotate
                if dir == 0:
                    rotated_kpt = kpt
                elif dir == 90:
                    img = cv2.imread(str(images_dir / path.name))
                    w, h = height, width    # reverse h, w, since img is rotated 90 degree
                    rotated_kpt = np.zeros_like(kpt)
                    rotated_kpt[:, 0] = kpt[:, 1]
                    rotated_kpt[:, 1] = w - 1 - kpt[:, 0]
                elif dir == 180:
                    img = cv2.imread(str(images_dir / path.name))
                    w, h = width, height
                    rotated_kpt = np.zeros_like(kpt)
                    rotated_kpt[:, 0] = w - 1 - kpt[:, 0]
                    rotated_kpt[:, 1] = h - 1 - kpt[:, 1]
                elif dir == 270:
                    img = cv2.imread(str(images_dir / path.name))
                    w, h = height, width    # reverse h, w, since img is rotated 90 degree
                    rotated_kpt = np.zeros_like(kpt)
                    rotated_kpt[:, 0] = h - 1 - kpt[:, 1]
                    rotated_kpt[:, 1] = kpt[:, 0]
                
                # Revert Crop
                rotated_kpt[:, 0] += rect[0]
                rotated_kpt[:, 1] += rect[1]

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
        with h5py.File(matches_h5_path, mode="r") as f_matches:
            for dir_key1 in f_matches.keys():
                dir1 = dir_key1.split(" ")[-1]
                key1 = " ".join(dir_key1.split(" ")[:-1])
                offset1 = offsets[key1][int(dir1)]
                matches[key1] = {}

                for dir_key2 in f_matches[dir_key1].keys():
                    dir2 = dir_key2.split(" ")[-1]
                    key2 = " ".join(dir_key2.split(" ")[:-1])
                    offset2 = offsets[key2][int(dir2)]

                    match = f_matches[dir_key1][dir_key2][...]
                    match[:, 0] += offset1
                    match[:, 1] += offset2
                    matches[key1][key2] = match
        
        matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])
        
        # Remove tmp dir
        shutil.rmtree(temp_dir)


    elif matcher_type == "DKM":
        # Matching
        images_dir = params["data_dict"][0].parent
        image_pairs = list(zip(image_pair_df["key1"].values.tolist(), image_pair_df["key2"].values.tolist()))
        dir_pairs = list(zip(image_pair_df["dir1"].values.tolist(), image_pair_df["dir2"].values.tolist()))
        keypoints, matches = keypoint_matcing_DKM(
                                image_pairs,
                                dir_pairs,
                                images_dir,
                                **params["keypoint_matching_args"],
                                rects=rects,
                                device=params["device"]
                             )
        
        # Save to h5
        keypoints_h5_path = os.path.join(work_dir, params["output"]["keypoints"])
        with h5py.File(keypoints_h5_path, mode="w") as f_keypoints:
            for k, v in keypoints.items():
                f_keypoints[k] = v
        
        matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])
        

    elif matcher_type == "LoFTR":
        # Matching
        images_dir = params["data_dict"][0].parent
        image_pairs = list(zip(image_pair_df["key1"].values.tolist(), image_pair_df["key2"].values.tolist()))
        dir_pairs = list(zip(image_pair_df["dir1"].values.tolist(), image_pair_df["dir2"].values.tolist()))
        keypoints, matches = keypoint_matcing_LoFTR(
                                image_pairs,
                                dir_pairs,
                                images_dir,
                                **params["keypoint_matching_args"],
                                rects=rects,
                                device=params["device"]
                             )
        
        # Save to h5
        keypoints_h5_path = os.path.join(work_dir, params["output"]["keypoints"])
        with h5py.File(keypoints_h5_path, mode="w") as f_keypoints:
            for k, v in keypoints.items():
                f_keypoints[k] = v
        
        matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])
    
    
    elif matcher_type == "Xfeat":
        # Matching
        images_dir = params["data_dict"][0].parent
        image_pairs = list(zip(image_pair_df["key1"].values.tolist(), image_pair_df["key2"].values.tolist()))
        dir_pairs = list(zip(image_pair_df["dir1"].values.tolist(), image_pair_df["dir2"].values.tolist()))
        keypoints, matches = keypoint_matcing_Xfeat(
                                image_pairs,
                                dir_pairs,
                                images_dir,
                                **params["keypoint_matching_args"],
                                rects=rects,
                                device=params["device"]
                             )
        
        # Save to h5
        keypoints_h5_path = os.path.join(work_dir, params["output"]["keypoints"])
        with h5py.File(keypoints_h5_path, mode="w") as f_keypoints:
            for k, v in keypoints.items():
                f_keypoints[k] = v
        
        matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])
    

    elif matcher_type == "LightGlue_pair":
        # Matching
        images_dir = params["data_dict"][0].parent
        image_pairs = list(zip(image_pair_df["key1"].values.tolist(), image_pair_df["key2"].values.tolist()))
        dir_pairs = list(zip(image_pair_df["dir1"].values.tolist(), image_pair_df["dir2"].values.tolist()))
        extractor_type = params["extractor"]
        keypoints, matches = keypoint_matcing_LG_pair(
                                image_pairs,
                                dir_pairs,
                                images_dir,
                                extractor_type,
                                keypoint_detection_args=params["keypoint_detection_args"],
                                keypoint_matching_args=params["keypoint_matching_args"],
                                rects=rects,
                                device=params["device"]
                             )
        
        # Save to h5
        keypoints_h5_path = os.path.join(work_dir, params["output"]["keypoints"])
        with h5py.File(keypoints_h5_path, mode="w") as f_keypoints:
            for k, v in keypoints.items():
                f_keypoints[k] = v
        
        matches_h5_path = os.path.join(work_dir, params["output"]["matches"])
        with h5py.File(matches_h5_path, mode="w") as f_matches:
            for key1 in matches.keys():
                for key2 in matches[key1].keys():
                    group  = f_matches.require_group(key1)
                    group.create_dataset(key2, data=matches[key1][key2])

    else:
        raise NotImplemented