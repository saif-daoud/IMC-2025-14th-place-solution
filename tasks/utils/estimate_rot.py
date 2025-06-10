import os
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import albumentations as albu
from timm import create_model as timm_create_model
import torch
from torch import nn
import re
import cv2
from typing import Any, Dict, Optional, Union

class DetectRotateImageWrapper:

    def __init__(self, weights, rot_threshold=0.8, debug=False):
        self.debug = debug
        self.rot_threshold = rot_threshold

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)
        self.model = self.create_model(weights)
        self.model = self.model.to(self.device).eval()

    def create_model(self, weights, activation: Optional[str] = "softmax") -> nn.Module:
        def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in state_dict.items():
                for key_r, value_r in rename_in_layers.items():
                    key = re.sub(key_r, value_r, key)
                result[key] = value
            return result

        model = timm_create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4)
        state_dict = torch.load(weights, map_location=self.device)['state_dict']
        state_dict = rename_layers(state_dict, {"model.": ""})
        model.load_state_dict(state_dict)
        if activation == "softmax":
            return nn.Sequential(model, nn.Softmax(dim=1))
        return model

    def tensor_from_rgb_image(self, image: np.ndarray) -> torch.Tensor:
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return torch.from_numpy(image)

    def load_rgb(self, image_path: Union[Path, str]) -> np.array:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_unrotated_img(self, image):
        x = self.transform(image=image)["image"]
        temp = [self.tensor_from_rgb_image(x)]
        with torch.no_grad():
            inp = torch.stack(temp).to(self.device)
            prediction = self.model(inp).detach().cpu().numpy()
        del temp, x, inp
        # [[2.6717133e-04 1.4841734e-04 2.5109938e-04 9.9933332e-01]]
        if self.debug:
            print(prediction)
        max_idx = np.argmax(prediction)
        max_pred = np.max(prediction)
        return max_idx, max_pred

    def de_rot_image(self, cv_image_rgb, ret_img=False):
        max_pred_idx, max_pred = self.detect_unrotated_img(cv_image_rgb)
        rotate_times = (4 - max_pred_idx) % 4
        is_rotated = rotate_times > 0 and max_pred >= self.rot_threshold  # 1,2 or 3
        if ret_img:
            im_unrotated = np.rot90(cv_image_rgb, rotate_times) if is_rotated else cv_image_rgb
            return is_rotated, rotate_times, max_pred, im_unrotated  # rgb image
        else:
            return is_rotated, rotate_times, max_pred

    def de_rot_image_path(self, im_path, ret_img=False):
        cv_image_rgb = self.load_rgb(im_path)
        return self.de_rot_image(cv_image_rgb, ret_img=ret_img)


def task_estimate_rot(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]

    rot_weights = os.path.join(params["input_dir_root"], params["rot_model_config"]["weight"])
    rot_model = DetectRotateImageWrapper(rot_weights, rot_threshold=params["rot_model_config"]["rot_thr"], debug=False)

    image_pair_df = pd.read_csv(os.path.join(work_dir, params["input"]))
    images_dir = params["data_dict"][0].parent
    img1_list = image_pair_df["key1"].values.tolist()
    img2_list = image_pair_df["key2"].values.tolist()
    image_paths = list(set(img1_list+img2_list))
    image_paths = [os.path.join(str(images_dir), p) for p in image_paths]

    rot_results = {}
    for image_path in tqdm(image_paths):
        is_rotated, rotate_times, conf = rot_model.de_rot_image_path(image_path)
        dig = [0, 270, 180, 90][rotate_times]
        rot_results[os.path.basename(image_path)] = dig
    
    dir1_list = []
    dir2_list = []
    for i, row in image_pair_df.iterrows():
        key1 = row["key1"]
        key2 = row["key2"]
        dir1_list.append(rot_results[key1])
        dir2_list.append(rot_results[key2])
    image_pair_df["dir1"] = dir1_list
    image_pair_df["dir2"] = dir2_list
    image_pair_df.to_csv(os.path.join(work_dir, params["output"]))

