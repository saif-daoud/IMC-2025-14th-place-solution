import os
import numpy as np
import h5py
import pandas as pd
import cv2
from tqdm import tqdm
from munkres import Munkres

def create_bg_mask(img):
    orig_shape = img.shape
    resized_img = cv2.resize(img, None, fx=0.1, fy=0.1)
    edges = cv2.Canny(resized_img, 35, 100)
    kernel_size = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 開閉操作を行う
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    bg_mask = cv2.drawContours(edges, [largest_contour], 0, 255, -1)
    bg_mask = cv2.resize(bg_mask, (orig_shape[1], orig_shape[0]))
    bg_mask = cv2.erode(bg_mask, kernel, iterations=5)
    return bg_mask

def create_high_reflection_mask(img, kernel_size=101, sigma=100, threshold_value=-1, bg_mask=None):
    blurred_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (kernel_size, kernel_size), sigmaX=sigma)
    
    if threshold_value < 0:
        h, w = blurred_img.shape
        zero_num = ((blurred_img&bg_mask)==0).sum()
        threshold_value = None
        for th in range(256):
            rate = ((blurred_img&bg_mask) > th).sum() / ((h*w)-zero_num)

            if rate < 0.45:
                threshold_value = th
                break
        
        if threshold_value is None:
            threshold_value = 0
    
    _, reflect_mask = cv2.threshold(blurred_img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return reflect_mask

def task_get_transparent_pair(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    image_dir = params["data_dict"][0].parent

    input_df = pd.read_csv(os.path.join(work_dir, params["input"]["image_pair"]))
    key_list = list(set(input_df["key1"].values.tolist() + input_df["key2"].values.tolist()))

    masks = {}
    for img_name in tqdm(key_list):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        bg_mask = create_bg_mask(img)
        high_reflection_mask = create_high_reflection_mask(img, threshold_value=-1, bg_mask=bg_mask)
        mask = bg_mask & high_reflection_mask
        masks[img_name] = mask
    
    keypoints_h5_path = os.path.join(work_dir, params["input"]["keypoints"])
    keypoints = {}
    with h5py.File(keypoints_h5_path, mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            keypoints[key] = f_keypoints[key][...]
    



    # # temp Add
    # matches_h5_path = os.path.join(work_dir, params["input"]["matches"])
    # matching_hists = {}
    # for target_key in key_list:
    #     target_keypoints = keypoints[target_key]
    #     matching_counter = np.zeros(target_keypoints.shape[0]).astype(np.int32)
    #     with h5py.File(matches_h5_path, mode="r") as f_matches:
    #         for key1 in f_matches.keys():
    #             for key2 in f_matches[key1].keys():
    #                 if key1 == target_key:
    #                     m = f_matches[key1][key2][...]
    #                     matching_counter[m[:, 0]] += 1
    #                 elif key2 == target_key:
    #                     m = f_matches[key1][key2][...]
    #                     matching_counter[m[:, 1]] += 1
    #     matching_hist = matching_counter / matching_counter.max()
    #     matching_hists[target_key] = matching_hist
    # ###########
    



    match_infos = {}
    for key in tqdm(key_list):
        match_infos[key] = {
             "ref_key": [],
             "valid_match_num": [],
             "valid_match_rate": [],
        }
    
    matches_h5_path = os.path.join(work_dir, params["input"]["matches"])
    with h5py.File(matches_h5_path, mode="r") as f_matches:
        for key1 in f_matches.keys():
            for key2 in f_matches[key1].keys():
                m = f_matches[key1][key2][...]
                cnt = 0
                for i in range(len(m)):
                    keypoint1 = keypoints[key1][m[i][0]]
                    keypoint2 = keypoints[key2][m[i][1]]

                    dist = np.linalg.norm(keypoint1-keypoint2)
                    if dist < 30 or dist > 200:
                        continue


                    mask1 = masks[key1]
                    x1, y1 = int(keypoint1[0]), int(keypoint1[1])
                    if not mask[y1, x1]:
                        continue

                    mask2 = masks[key2]
                    x2, y2 = int(keypoint2[0]), int(keypoint2[1])
                    if not mask[y2, x2]:
                        continue

                    # matching_hist1 = matching_hists[key1]
                    # matching_hist2 = matching_hists[key2]
                    # if matching_hist1[m[i][0]] > 0.1 or matching_hist2[m[i][1]] > 0.2:
                    #     continue

                    cnt += 1
                
                match_infos[key1]["ref_key"].append(key2)
                match_infos[key1]["valid_match_num"].append(cnt)
                match_infos[key1]["valid_match_rate"].append(cnt/len(m))

                match_infos[key2]["ref_key"].append(key1)
                match_infos[key2]["valid_match_num"].append(cnt)
                match_infos[key2]["valid_match_rate"].append(cnt/len(m))

    


    # m = Munkres()
    # mat = np.ones((len(key_list), len(key_list)))
    # for target_key in key_list:
    #     match_info = match_infos[target_key]
    #     for ref_key, valid_match_num, valid_match_rate in zip(match_info["ref_key"], match_info["valid_match_num"], match_info["valid_match_rate"]):
    #         if valid_match_num > 100 and valid_match_rate > 0.3:
    #             i = key_list.index(target_key)
    #             j = key_list.index(ref_key)
    #             mat[i][j] = -1 * valid_match_rate**2 * valid_match_num
    #             mat[j][i] = -1 * valid_match_rate**2 * valid_match_num

    # temp = m.compute(mat)
    # for t in temp:
    #     i, j = t
    #     cor = "o" if (j, i) in temp else "x"
    #     print(f"[{cor}] {key_list[i]} - {key_list[j]}")
    # import pdb;pdb.set_trace()




    # extract pair
    all_pair_list = []
    for key, value in match_infos.items():
        pair_list = []
        registerd_ref_keys = []

        # matching_scores = np.array(value["valid_match_rate"]) * (np.array(value["valid_match_num"]) / np.array(value["valid_match_num"]).max())
        # matching_scores = matching_scores.tolist()
        # sorted_pairs = sorted(zip(matching_scores, value["ref_key"], value["valid_match_num"], value["valid_match_rate"]), key=lambda x:x[0], reverse=True)
        # sorted_matching_scores, sorted_ref_keys, valid_match_nums, valid_match_rates  = zip(*sorted_pairs)
        # for i in range(params["k"]):
        #     pair_list.append((key, sorted_ref_keys[i]))


        # 1st search (sufficient_valid_num & sufficient_valid_rate)
        sorted_pairs = sorted(zip(value["valid_match_num"], value["valid_match_rate"], value["ref_key"]), key=lambda x:x[0], reverse=True)   # sort valid_num
        valid_match_nums, valid_match_rates, ref_keys = zip(*sorted_pairs)
        for valid_match_num, valid_match_rate, ref_key in zip(valid_match_nums, valid_match_rates, ref_keys):
            if valid_match_num > 100 and valid_match_rate > 0.4 and ref_key not in registerd_ref_keys:
                pair_list.append((key, ref_key))
                registerd_ref_keys.append(ref_key)
                if len(pair_list) == params["k"]:
                    break
                if valid_match_num <= 100:
                    break
        
        # 2nd search (reasonable_valid_num)
        if len(pair_list) < params["k"]:
            sorted_pairs = sorted(zip(value["valid_match_num"], value["valid_match_rate"], value["ref_key"]), key=lambda x:x[1], reverse=True)   # sort valid_rate
            valid_match_nums, valid_match_rates, ref_keys = zip(*sorted_pairs)
            for valid_match_num, valid_match_rate, ref_key in zip(valid_match_nums, valid_match_rates, ref_keys):
                if valid_match_num > 50 and ref_key not in registerd_ref_keys:
                    pair_list.append((key, ref_key))
                    registerd_ref_keys.append(ref_key)

                    if len(pair_list) == params["k"]:
                        break
                    if valid_match_num <= 50:
                        break
        
        # 3rd search 
        if len(pair_list) < params["k"]:
            sorted_pairs = sorted(zip(value["valid_match_num"], value["valid_match_rate"], value["ref_key"]), key=lambda x:x[0], reverse=True)   # sort valid_num
            valid_match_nums, valid_match_rates, ref_keys = zip(*sorted_pairs)
            for valid_match_num, valid_match_rate, ref_key in zip(valid_match_nums, valid_match_rates, ref_keys):
                if ref_key not in registerd_ref_keys:
                    pair_list.append((key, ref_key))
                    registerd_ref_keys.append(ref_key)

                    if len(pair_list) == params["k"]:
                        break
        
        all_pair_list += pair_list
        import pdb;pdb.set_trace()

    #import pdb;pdb.set_trace()
    dst_data_dict = {
        "key1": [],
        "key2": [],
        "sim": [],
        "dir1": [],
        "dir2": [],
        "match_num": []
    }
    for i, row in input_df.iterrows():
        pair = (row["key1"], row["key2"])
        rev_pair = (row["key2"], row["key1"])
        if pair not in all_pair_list and rev_pair not in all_pair_list:
            continue

        dst_data_dict["key1"].append(row["key1"])
        dst_data_dict["key2"].append(row["key2"])
        dst_data_dict["sim"].append(row["sim"])
        dst_data_dict["dir1"].append(row["dir1"])
        dst_data_dict["dir2"].append(row["dir2"])
        dst_data_dict["match_num"].append(row["match_num"])

    dst_df = pd.DataFrame.from_dict(dst_data_dict)
    print(f"pair_num = {len(dst_df)}")

    output_path = os.path.join(work_dir, params["output"])
    print(f"save -> {output_path}")
    dst_df.to_csv(output_path, index=False)