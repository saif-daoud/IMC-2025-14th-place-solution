from tqdm import tqdm
from pathlib import Path
import h5py
import numpy as np
import torch
import kornia as K
import kornia.feature as KF

try:
    from dkm import DKMv3_outdoor
except:
    print("DKM is not installed.Please install it if you want to use it.")
from kornia.feature import LoFTR
from tasks.matching.core.accelerated_features.modules.xfeat import XFeat
from PIL import Image
import cv2

def keypoint_matcing_LG(
    image_pairs: list[tuple[str, str]],
    keypoints_h5_path: str,
    descriptions_h5_path: str,
    extractor_type: str,
    matcher_params: dict,
    min_matches: int = 15,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Computes distances between keypoints of images.
    
    Stores output at feature_dir/matches.h5
    """

    matcher = KF.LightGlueMatcher(extractor_type, matcher_params).eval().to(device)
    matches = {}
    with h5py.File(keypoints_h5_path, mode="r") as f_keypoints, \
         h5py.File(descriptions_h5_path, mode="r") as f_descriptors:
            for key1, key2 in tqdm(image_pairs, desc="Computing keypoing distances"):
                keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
                keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
                descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
                descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)

                with torch.inference_mode():
                    distances, indices = matcher(
                        descriptors1, 
                        descriptors2, 
                        KF.laf_from_center_scale_ori(keypoints1[None]),
                        KF.laf_from_center_scale_ori(keypoints2[None]),
                    )
                
                # We have matches to consider
                n_matches = len(indices)
                if n_matches:
                    if verbose:
                        print(f"{key1}-{key2}: {n_matches} matches")
                    # Store the matches in the group of one image
                    if n_matches >= min_matches:
                        matches.setdefault(key1, {})
                        matches[key1][key2] = indices.detach().cpu().numpy().reshape(-1, 2)
    return matches


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


def keypoint_matcing_DKM(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    sample_thresh: float = 0.5,
    img_size: list = None,
    max_matching_num: int = 10000,
    min_matches: int = 15,
    rects: dict = None,
    device: torch.device = torch.device("cpu"),
):
    dkm_model = DKMv3_outdoor(device=device)
    dkm_model.sample_thresh = sample_thresh
    if img_size is not None:
        dkm_model.h_resized = img_size[0]
        dkm_model.w_resized = img_size[1]
        dkm_model.upsample_preds = False
    
    keypoints = {}
    matches = {}
    for pair, dir_pair in tqdm(zip(image_pairs, dir_pairs), total=len(image_pairs)):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        img1 = cv2.imread(str(images_dir / key1), cv2.IMREAD_COLOR) 
        img2 = cv2.imread(str(images_dir / key2), cv2.IMREAD_COLOR)

        # Crop
        if rects is not None and key1 in rects:
            # sfm mkpc
            if type(rects[key1]) == list:
                rect1 = rects[key1]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect1 = rects[key1][key2]
            img1 = img1[rect1[1]:rect1[3], rect1[0]:rect1[2], :]
        else:
            rect1 = [0, 0, 0, 0]
        
        if rects is not None and key2 in rects:
            # sfm mkpc
            if type(rects[key2]) == list:
                rect2 = rects[key2]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect2 = rects[key2][key1]
            img2 = img2[rect2[1]:rect2[3], rect2[0]:rect2[2], :]
        else:
            rect2 = [0, 0, 0, 0]

        # Rotate
        if dir1==90:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif dir1==180:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif dir1==270:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if dir2==90:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        elif dir2==180:
            img2 = cv2.rotate(img2, cv2.ROTATE_180)
        elif dir2==270:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        
        # Original (https://github.com/Parskatt/DKM/blob/main/demo/demo_fundamental.py)
        warp, certainty = dkm_model.match(img1PIL, img2PIL)
        W_A, H_A = img1PIL.size
        W_B, H_B = img2PIL.size
        dense_matches, certainty = dkm_model.sample(warp, certainty, num=max_matching_num)
        kpts1, kpts2 = dkm_model.to_pixel_coordinates(dense_matches, H_A, W_A, H_B, W_B)
        kpts1 = kpts1.detach().cpu().numpy()
        kpts2 = kpts2.detach().cpu().numpy()
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


def convert_image_LoFTR(img, size, device):
    original_shape = img.shape
    w = size[1]
    h = size[0]
    img_resized = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img_resized, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device), original_shape

def keypoint_matcing_LoFTR(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    sample_thresh: float = 0.5,
    img_size: list = [840, 840],
    min_matches: int = 15,
    rects: dict = None,
    device: torch.device = torch.device("cpu"),
):
    loftr_model = LoFTR().eval().to(device)
    keypoints = {}
    matches = {}
    for pair, dir_pair in tqdm(zip(image_pairs, dir_pairs), total=len(image_pairs)):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        img1 = cv2.imread(str(images_dir / key1), cv2.IMREAD_COLOR) 
        img2 = cv2.imread(str(images_dir / key2), cv2.IMREAD_COLOR)

        # Crop
        if rects is not None and key1 in rects:
            # sfm mkpc
            if type(rects[key1]) == list:
                rect1 = rects[key1]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect1 = rects[key1][key2]
            img1 = img1[rect1[1]:rect1[3], rect1[0]:rect1[2], :]
        else:
            rect1 = [0, 0, 0, 0]
        
        if rects is not None and key2 in rects:
            # sfm mkpc
            if type(rects[key2]) == list:
                rect2 = rects[key2]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect2 = rects[key2][key1]
            img2 = img2[rect2[1]:rect2[3], rect2[0]:rect2[2], :]
        else:
            rect2 = [0, 0, 0, 0]

        # Rotate
        if dir1==90:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif dir1==180:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif dir1==270:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if dir2==90:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        elif dir2==180:
            img2 = cv2.rotate(img2, cv2.ROTATE_180)
        elif dir2==270:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Get image size
        H_A, W_A, _ = img1.shape
        H_B, W_B, _ = img2.shape


        # Convert image
        resized_image_1, ori_shape_1 = convert_image_LoFTR(img1, img_size, device)
        resized_image_2, ori_shape_2 = convert_image_LoFTR(img2, img_size, device)
        input_dict = {"image0": K.color.rgb_to_grayscale(resized_image_1), 
                      "image1": K.color.rgb_to_grayscale(resized_image_2)}

        with torch.no_grad():
            correspondences = loftr_model(input_dict)
        kpts1 = correspondences['keypoints0'].cpu().numpy()
        kpts2 = correspondences['keypoints1'].cpu().numpy()
        mconf  = correspondences['confidence'].cpu().numpy()
        
        kpts1 = kpts1[ mconf >= sample_thresh, : ]
        kpts2 = kpts2[ mconf >= sample_thresh, : ]
        mconf  = mconf[ mconf >= sample_thresh ]

        # Scaling coords to same pixel size of LoFTR
        kpts1[:,0] = kpts1[:,0] * ori_shape_1[1] / resized_image_1.shape[3]   # X
        kpts1[:,1] = kpts1[:,1] * ori_shape_1[0] / resized_image_1.shape[2]   # Y
        kpts2[:,0] = kpts2[:,0] * ori_shape_2[1] / resized_image_2.shape[3]   # X
        kpts2[:,1] = kpts2[:,1] * ori_shape_2[0] / resized_image_2.shape[2]   # Y

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

        # save result
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




def resize_img(img, img_size):
    if type(img_size) == list:
        resized_image = cv2.resize(img, (img_size[0], img_size[1]))
    else:
        h, w = img.shape[:2]
        if h > w:
            long_edge = h
            scale = img_size / h
        else:
            long_edge = w
            scale = img_size / w
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = cv2.resize(img, (new_w, new_h))
    return resized_image

def Xfeat_get_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return keypoints1, keypoints2, matches

def keypoint_matcing_Xfeat(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    top_k: float = 0.5,
    sparse:bool = True,
    img_size: list = [840, 840],
    min_matches: int = 15,
    rects: dict = None,
    device: torch.device = torch.device("cpu"),
):
    model = XFeat(device=device)
    keypoints = {}
    matches = {}
    for pair, dir_pair in tqdm(zip(image_pairs, dir_pairs), total=len(image_pairs)):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        img1 = cv2.imread(str(images_dir / key1), cv2.IMREAD_COLOR) 
        img2 = cv2.imread(str(images_dir / key2), cv2.IMREAD_COLOR)

        # Crop
        if rects is not None and key1 in rects:
            # sfm mkpc
            if type(rects[key1]) == list:
                rect1 = rects[key1]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect1 = rects[key1][key2]
            img1 = img1[rect1[1]:rect1[3], rect1[0]:rect1[2], :]
        else:
            rect1 = [0, 0, 0, 0]
        
        if rects is not None and key2 in rects:
            # sfm mkpc
            if type(rects[key2]) == list:
                rect2 = rects[key2]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect2 = rects[key2][key1]
            img2 = img2[rect2[1]:rect2[3], rect2[0]:rect2[2], :]
        else:
            rect2 = [0, 0, 0, 0]

        # Rotate
        if dir1==90:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif dir1==180:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif dir1==270:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if dir2==90:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        elif dir2==180:
            img2 = cv2.rotate(img2, cv2.ROTATE_180)
        elif dir2==270:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Resize
        ori_shape_1 = img1.shape
        resized_image_1 = resize_img(img1, img_size)
        ori_shape_2 = img2.shape
        resized_image_2 = resize_img(img2, img_size)

        # Matching
        if sparse:
            kpts1, kpts2 = model.match_xfeat(resized_image_1, resized_image_2, top_k = top_k)
        else:
            kpts1, kpts2 = model.match_xfeat_star(resized_image_1, resized_image_2, top_k = top_k)
        H, mask = cv2.findHomography(kpts1, kpts2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        mask = mask.flatten()
        kpts1 = kpts1[mask==1]
        kpts2 = kpts2[mask==1]

        # Scaling coords to same pixel size of LoFTR
        kpts1[:,0] = kpts1[:,0] * ori_shape_1[1] / resized_image_1.shape[1]   # X
        kpts1[:,1] = kpts1[:,1] * ori_shape_1[0] / resized_image_1.shape[0]   # Y
        kpts2[:,0] = kpts2[:,0] * ori_shape_2[1] / resized_image_2.shape[1]   # X
        kpts2[:,1] = kpts2[:,1] * ori_shape_2[0] / resized_image_2.shape[0]   # Y

        if kpts1.shape[0] < min_matches or kpts2.shape[0]< min_matches:
            continue
        
        # Revert Rotate
        H_A, W_A, _ = ori_shape_1
        H_B, W_B, _ = ori_shape_2
        kpts1 = revert_rotate(kpts1, dir1, W_A, H_A)
        kpts2 = revert_rotate(kpts2, dir2, W_B, H_B)

        # Revert Crop
        kpts1[:, 0] += rect1[0]
        kpts1[:, 1] += rect1[1]
        kpts2[:, 0] += rect2[0]
        kpts2[:, 1] += rect2[1]

        # save result
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



if __name__=='__main__':
    import itertools
    images_list = list(Path("../datas/input/image-matching-challenge-2024/test/church/images/").glob("*.png"))[:10]
    feature_dir = Path("./sample_test_features")
    index_pairs = list(itertools.combinations(range(len(images_list)), 2))
    keypoint_matcing_LG(images_list, index_pairs, feature_dir, verbose=False)