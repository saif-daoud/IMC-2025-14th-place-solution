import os
from tqdm import tqdm
from pathlib import Path
import h5py
import numpy as np
import torch
from lightglue import ALIKED, DISK, SIFT, SuperPoint
import kornia as K
import kornia.feature as KF
from PIL import Image
import cv2

extractor_map = {
    "aliked": ALIKED,
    "disk": DISK,
    "sift": SIFT,
    "superpoint": SuperPoint,
}

input_dim_map = {
    "aliked": 128,
    "disk": 128,
    "sift": 128,
    "superpoint": 256,
}

def load_lg_img(path, device, dtype, rect, direction):
    image = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)[None, ...].to(dtype)

    # Crop
    image = image[:,:,rect[1]:rect[3], rect[0]:rect[2]]

    # Rotate
    if direction==90:
        image = torch.rot90(image, k=-1, dims=(2,3))
    elif direction==180:
        image = torch.rot90(image, k=2, dims=(2,3))
    elif direction==270:
        image = torch.rot90(image, k=1, dims=(2,3))
    
    return image

def revert_rotate(kpt, direction, width, height):
    if direction == 0:
        rotated_kpt = kpt
    elif direction == 90:
        rotated_kpt = np.zeros_like(kpt)
        rotated_kpt[:, 0] = kpt[:, 1]
        rotated_kpt[:, 1] = width - 1 - kpt[:, 0]
    elif direction == 180:
        rotated_kpt = np.zeros_like(kpt)
        rotated_kpt[:, 0] = width - 1 - kpt[:, 0]
        rotated_kpt[:, 1] = height - 1 - kpt[:, 1]
    elif direction == 270:
        rotated_kpt = np.zeros_like(kpt)
        rotated_kpt[:, 0] = height - 1 - kpt[:, 1]
        rotated_kpt[:, 1] = kpt[:, 0]
    return rotated_kpt


def keypoint_matcing_LG_pair(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    extractor_type: str,
    keypoint_detection_args: dict,
    keypoint_matching_args: dict,
    rects: dict,
    device: torch.device = torch.device("cpu"),
):
    dtype = torch.float32
    if "dtype" in keypoint_detection_args:
        dtype = keypoint_detection_args["dtype"]
        if dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32 # ALIKED has issues with float16
    
    extractor_conf = keypoint_detection_args["extractor_conf"]
    preprocess_conf = {}
    if "preprocess_conf" in keypoint_detection_args:
        preprocess_conf = keypoint_detection_args["preprocess_conf"]
    extractor = extractor_map[extractor_type](**extractor_conf).eval().to(device, dtype)
    extractor.preprocess_conf.update(preprocess_conf)
    matcher = KF.LightGlueMatcher(extractor_type, keypoint_matching_args["matcher_params"]).eval().to(device)
    
    min_matches = keypoint_matching_args["min_matches"]
    keypoints = {}
    matches = {}
    for pair, dir_pair in tqdm(zip(image_pairs, dir_pairs), total=len(image_pairs)):
        key1, key2 = pair
        dir1, dir2 = dir_pair
        rect1 = rects[key1][key2]
        rect2 = rects[key2][key1]

        img_path1 = os.path.join(images_dir, key1)
        img_path2 = os.path.join(images_dir, key2)
        with torch.inference_mode():
            # compute keypoints
            img1 = load_lg_img(img_path1, device, dtype, rect1, dir1)
            features1 = extractor.extract(img1)
            keypoints1 = features1["keypoints"].squeeze().detach().cpu().numpy()
            descriptors1 = features1["descriptors"].squeeze().detach().cpu().numpy()

            img2 = load_lg_img(img_path2, device, dtype, rect2, dir2)
            features2 = extractor.extract(img2)
            keypoints2 = features2["keypoints"].squeeze().detach().cpu().numpy()
            descriptors2 = features2["descriptors"].squeeze().detach().cpu().numpy()

            _, _, H_A, W_A = img1.shape
            _, _, H_B, W_B = img2.shape

            # Check Shape
            input_dim = input_dim_map[extractor_type]
            keypoints1 = keypoints1.reshape((-1, 2))
            descriptors1 = descriptors1.reshape((-1, input_dim))
            keypoints2 = keypoints2.reshape((-1, 2))
            descriptors2 = descriptors2.reshape((-1, input_dim))

            # Check range
            mask1 = (keypoints1[:,0] >= 0) \
                    & (keypoints1[:,0] < img1.shape[3]) \
                    & (keypoints1[:,1] >= 0) \
                    & (keypoints1[:,1] < img1.shape[2])
            keypoints1 = keypoints1[mask1]
            descriptors1 = descriptors1[mask1]

            mask2 = (keypoints2[:,0] >= 0) \
                    & (keypoints2[:,0] < img2.shape[3]) \
                    & (keypoints2[:,1] >= 0) \
                    & (keypoints2[:,1] < img2.shape[2])
            keypoints2 = keypoints2[mask2]
            descriptors2 = descriptors2[mask2]

            del img1
            del img2

            # matching
            keypoints1_tensor = torch.from_numpy(keypoints1).to(device)
            keypoints2_tensor = torch.from_numpy(keypoints2).to(device)
            descriptors1_tensor = torch.from_numpy(descriptors1).to(device)
            descriptors2_tensor = torch.from_numpy(descriptors2).to(device)
            distances, indices = matcher(
                descriptors1_tensor, 
                descriptors2_tensor, 
                KF.laf_from_center_scale_ori(keypoints1_tensor[None]),
                KF.laf_from_center_scale_ori(keypoints2_tensor[None]),
            )

            m = indices.detach().cpu().numpy().reshape(-1, 2)
            kpts1 = keypoints1[m[:, 0]]
            kpts2 = keypoints2[m[:, 1]]

        if kpts1.shape[0] < min_matches or kpts2.shape[0]< min_matches:
            continue

        # Revert Rotate
        kpts1 = revert_rotate(kpts1, dir1, W_A, H_A)
        kpts2 = revert_rotate(kpts2, dir2, W_B, H_B)

        # Revert Crop
        kpts1[:, 0] += rect1[0]
        kpts1[:, 1] += rect1[1]
        kpts2[:, 0] += rect2[0]
        kpts2[:, 1] += rect2[1]


        if key1 not in keypoints:
            keypoints[key1] = kpts1
            matches1 = np.array(list(range(0, kpts1.shape[0])))
        else:
            n = keypoints[key1].shape[0]
            keypoints[key1] = np.concatenate([keypoints[key1], kpts1])
            matches1 = np.array(list(range(0, kpts1.shape[0]))) + n
        
        if key2 not in keypoints:
            keypoints[key2] = kpts2
            matches2 = np.array(list(range(0, kpts2.shape[0])))
        else:
            n = keypoints[key2].shape[0]
            keypoints[key2] = np.concatenate([keypoints[key2], kpts2])
            matches2 = np.array(list(range(0, kpts2.shape[0]))) + n

        _matches = np.zeros((matches1.shape[0], 2)).astype(np.int64)
        _matches[:, 0] = matches1
        _matches[:, 1] = matches2
        matches.setdefault(key1, {})
        matches[key1][key2] = _matches
    
    return keypoints, matches

    