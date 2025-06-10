import os
import json
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def task_transparent_crop(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    images_dir = params["data_dict"][0].parent
    image_pair_df = pd.read_csv(os.path.join(work_dir, params["input"]))
    img1_list = image_pair_df["key1"].values.tolist()
    img2_list = image_pair_df["key2"].values.tolist()
    img_list = list(set(img1_list+img2_list))

    results = {}
    for img_name in tqdm(img_list):
        img = cv2.imread(os.path.join(images_dir, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=0.1, fy=0.1)
        edges = cv2.Canny(img, 50, 150)
        points = np.argwhere(edges > 0)
        if points.size == 0:
            rect = [0, 0, img.shape[1], img.shape[0]]
        else:        
            top_left = np.min(points, axis=0)
            bottom_right = np.max(points, axis=0)
            x1, y1 = top_left[::-1]
            x2, y2 = bottom_right[::-1]
            rect = [int(x1)*10, int(y1)*10, int(x2)*10, int(y2)*10]
        results[img_name] = rect

    # Save to JSON
    output_path = os.path.join(work_dir, params["output"])
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Save to {output_path}")