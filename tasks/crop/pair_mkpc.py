import os
import numpy as np
import h5py
import cv2
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


def task_pair_mkpc(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    image_dir = params["data_dict"][0].parent

    keypoints = {}
    with h5py.File(os.path.join(work_dir, params["input"]["keypoints"]), mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            keypoints[key] = f_keypoints[key][...]
    
    results = {}
    with h5py.File(os.path.join(work_dir, params["input"]["matches"]), mode="r") as f_matches:
        for key1 in f_matches.keys():
            for key2 in f_matches[key1].keys():
                matches = f_matches[key1][key2][...]
                kpts1 = keypoints[key1][matches[:, 0]]
                kpts2 = keypoints[key2][matches[:, 1]]

                # Normalize mkpt
                img1 = cv2.imread(os.path.join(image_dir, key1))
                img2 = cv2.imread(os.path.join(image_dir, key2))
                normalize_kpts1 = kpts1.copy()
                normalize_kpts1[:, 0] /= img1.shape[1]
                normalize_kpts1[:, 1] /= img1.shape[0]
                normalize_kpts2 = kpts2.copy()
                normalize_kpts2[:, 0] /= img2.shape[1]
                normalize_kpts2[:, 1] /= img2.shape[0]

                # DBSCAN
                db_scan_param = {
                    "eps": 0.05,
                    "min_samples": 5,
                    "metric": 'euclidean'
                }
                if "db_scan_param" in params:
                    db_scan_param.update(params["db_scan_param"])
                cls_pred1 = DBSCAN(**db_scan_param).fit_predict(normalize_kpts1)
                cls_pred2 = DBSCAN(**db_scan_param).fit_predict(normalize_kpts2)

                # MKPC
                rect1 = get_mkpt_rect(kpts1, cls_pred1, img1.shape, **params["mkpt_rect_parms"])
                rect2 = get_mkpt_rect(kpts2, cls_pred2, img2.shape, **params["mkpt_rect_parms"])
                
                if key1 not in results:
                    results[key1] = {}
                if key2 not in results:
                    results[key2] = {}
                results[key1][key2] = rect1
                results[key2][key1] = rect2
    
    # Save to JSON
    output_path = os.path.join(work_dir, params["output"])
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Save to {output_path}")