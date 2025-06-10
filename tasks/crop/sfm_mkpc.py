import os
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import json
import collections
from sklearn.cluster import DBSCAN

def get_mkpt_rect(mkpts, cls_pred, img_shape, crop_scale=[1.0, 1.0], ratio_thres=0.05, inliner_rate_thresh=0.5): 
    cls_count = collections.Counter(cls_pred)
    if -1 in cls_count:
        inliner_num = len(cls_pred) - cls_count[-1]
        inliner_rate = inliner_num/len(cls_pred)
    else:
        inliner_num = len(cls_pred)
        inliner_rate = 1.0
        
    # inlinerが少なすぎるなら、信頼性が低いのでクロップしない
    if inliner_rate < inliner_rate_thresh:
        rect = [0, 0, img_shape[1]-1, img_shape[0]-1]
        return rect
    # remove outlier class
    if -1 in cls_count:
        del cls_count[-1]
    # 上位以上の点が含まれるクラスタを選定
    sorted_cls_count = sorted(cls_count.items(), key=lambda x:x[1], reverse=True)
    sorted_cls_count_dict = {}
    for c, n in sorted_cls_count:
        sorted_cls_count_dict[c]=n
    # Get the size of the cluster containing the most matching points (ignoring outlier clusters)
    valid_cluster = []
    for class_name, n_item in sorted_cls_count:
        if len(valid_cluster) == 0:
            valid_cluster.append(class_name)
        else:
            ratio = n_item / sorted_cls_count_dict[valid_cluster[-1]]
            if ratio < ratio_thres:
                break
            valid_cluster.append(class_name)
            
    # マスクを作成して、mkptの最大・最小を取得
    mask = np.zeros(cls_pred.shape, dtype=bool)
    for vc in valid_cluster:
        mask = mask | (cls_pred==vc)
    valid_mkpts = mkpts[mask]
    lu_x = valid_mkpts[:, 0].min()
    lu_y = valid_mkpts[:, 1].min()
    rd_x = valid_mkpts[:, 0].max()
    rd_y = valid_mkpts[:, 1].max()
    
    center_x = (lu_x + rd_x) / 2.0
    center_y = (lu_y + rd_y) / 2.0
    
    crop_w = (rd_x - lu_x)*crop_scale[0]
    crop_h = (rd_y - lu_y)*crop_scale[1]
    
    lu_x = max(center_x - crop_w / 2, 0)
    lu_y = max(center_y - crop_h / 2, 0)
    rd_x = lu_x + crop_w
    rd_y = lu_y + crop_h
    
    rect = [max(0, int(lu_x)), max(0, int(lu_y)), min(int(rd_x), img_shape[1]-1), min(int(rd_y), img_shape[0]-1)]
    return rect


def task_sfm_mkpc(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]

    keypoints = {}
    with h5py.File(os.path.join(work_dir, params["input"]["keypoints"]), mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            keypoints[key] = f_keypoints[key][...]
    
    results = {}
    for target_key in tqdm(keypoints.keys()):
        image_path = os.path.join(params["data_dict"][0].parent, target_key)
        with Image.open(image_path) as img:
            width, height =  img.size
        img_shape = (height, width)
        
        target_keypoints = keypoints[target_key]
        matching_counter = np.zeros(target_keypoints.shape[0]).astype(np.int32)
        with h5py.File(os.path.join(work_dir, params["input"]["matches"]), mode="r") as f_matches:
            for key1 in f_matches.keys():
                for key2 in f_matches[key1].keys():
                    if key1 == target_key:
                        m = f_matches[key1][key2][...]
                        matching_counter[m[:, 0]] += 1
                    elif key2 == target_key:
                        m = f_matches[key1][key2][...]
                        matching_counter[m[:, 1]] += 1

        matching_hist = matching_counter / matching_counter.max()

        # Extract sufficient matching_num keypoints
        matched_keypoints = target_keypoints[matching_hist>params["thresh"]] 
        matching_hist = matching_hist[matching_hist>params["thresh"]]

        if matched_keypoints.shape[0] == 0:
            rect = [0, 0, img_shape[1]-1, img_shape[0]-1]
            results[target_key] = rect
            continue

        # Normalize mkpt
        normalize_matched_keypoints = matched_keypoints.copy()
        normalize_matched_keypoints[:, 0] /= img_shape[1]
        normalize_matched_keypoints[:, 1] /= img_shape[0]

        # DBSCAN
        np.random.seed(42)
        db_scan_param = {
            "eps": 0.05,
            "min_samples": 16,
            "metric": 'euclidean'
        }
        if "db_scan_param" in params:
            db_scan_param.update(params["db_scan_param"])
        #cls_pred = DBSCAN(eps=30, min_samples=8, metric='euclidean').fit_predict(matched_keypoints)
        cls_pred = DBSCAN(**db_scan_param).fit_predict(normalize_matched_keypoints)

        # sfm MKPC
        rect = get_mkpt_rect(matched_keypoints, cls_pred, img_shape, **params["mkpt_rect_parms"])
        results[target_key] = rect
    
    # Save to JSON
    output_path = os.path.join(work_dir, params["output"])
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Save to {output_path}")