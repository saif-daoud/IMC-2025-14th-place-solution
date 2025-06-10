from tqdm import tqdm
from pathlib import Path
import torch
from lightglue import ALIKED, DISK, SIFT, SuperPoint
import h5py
import kornia as K
from types import SimpleNamespace
from tasks.matching.core.superpoint_pytorch import *

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

def detect_keypoints(
    paths: list[Path],
    extractor_type: str,
    extractor_conf: dict,
    preprocess_conf: dict = {},
    rects: dict = None,
    dynamic_resize: list = None,
    dtype: str = "float32",
    device: torch.device = torch.device("cpu"),
) -> None:
    """Detects the keypoints in a list of images with ALIKED
    
    Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
    to be used later with LightGlue
    """
    if dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32 # ALIKED has issues with float16
    
    extractor = extractor_map[extractor_type](**extractor_conf).eval().to(device, dtype)
    extractor.preprocess_conf.update(preprocess_conf)

    keypoints = {}
    descriptors = {}
    for data in tqdm(paths, desc="Computing keypoints"):
        if type(data) == tuple:
            path = data[0]
            direction = data[1]
        else:
            path = data
            direction = -1
        
        key = path.name
        if direction != -1:
            key = key + f" {direction}"

        with torch.inference_mode():
            image = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)[None, ...].to(dtype)

            # Crop
            if rects is not None and path.name in rects:
                rect = rects[path.name]
                image = image[:,:,rect[1]:rect[3], rect[0]:rect[2]]

            # Rotate
            if direction==90:
                image = torch.rot90(image, k=-1, dims=(2,3))
            elif direction==180:
                image = torch.rot90(image, k=2, dims=(2,3))
            elif direction==270:
                image = torch.rot90(image, k=1, dims=(2,3))
            
            # dynamic_resize
            if dynamic_resize is not None:
                _, _, height, width = image.shape
                max_edge = max(height, width)
                resize = min(dynamic_resize, key=lambda x:abs(x-max_edge))
                extractor.preprocess_conf.update({"resize": resize})

            features = extractor.extract(image)
            
            keypoints[key] = features["keypoints"].squeeze().detach().cpu().numpy()
            descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()

            # Check Shape
            input_dim = input_dim_map[extractor_type]
            keypoints[key] = keypoints[key].reshape((-1, 2))
            descriptors[key] = descriptors[key].reshape((-1, input_dim))

            # Check range
            mask = (keypoints[key][:,0] >= 0) \
                    & (keypoints[key][:,0] < image.shape[3]) \
                    & (keypoints[key][:,1] >= 0) \
                    & (keypoints[key][:,1] < image.shape[2])
            keypoints[key] = keypoints[key][mask]
            descriptors[key] = descriptors[key][mask]

    return keypoints, descriptors
    

def detect_keypoints_superpoint(
    paths: list[Path],
    extractor_conf: dict,
    preprocess_conf: dict,
    dtype: str = "float32",
    device: torch.device = torch.device("cpu"),
) -> None:
    if dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32 # ALIKED has issues with float16
    
    default_preprocess_conf = {
        "resize": 1024,
        "side": "long",
        "align_corners": None,
        "antialias": True,
    }
    default_preprocess_conf.update(preprocess_conf)
    preprocess_conf = SimpleNamespace(**default_preprocess_conf)

    extractor = SuperPoint_open(**extractor_conf).eval().to(device, dtype)

    keypoints = {}
    descriptors = {}
    for path in tqdm(paths, desc="Computing keypoints"):
        key = path.name
        with torch.inference_mode():
            # image loading
            img = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)[None, ...].to(dtype)
            if img.dim() == 3:
                img = img[None]  # add batch dim
            assert img.dim() == 4 and img.shape[0] == 1
            shape = img.shape[-2:][::-1]

            # image preprocess
            h, w = img.shape[-2:]
            if preprocess_conf.resize is not None:
                img = K.geometry.transform.resize(
                    img,
                    preprocess_conf.resize,
                    side=preprocess_conf.side,
                    antialias=preprocess_conf.antialias,
                    align_corners=preprocess_conf.align_corners,
                )
            scales = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)

            # extract
            feats = extractor.forward({"image": img})
            feats["keypoints"] = feats["keypoints"][0]
            feats["keypoint_scores"] = feats["keypoint_scores"][0]
            feats["descriptors"] = feats["descriptors"][0]
            feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5

            # save result
            keypoints[key] = feats["keypoints"].squeeze().detach().cpu().numpy()
            descriptors[key] = feats["descriptors"].squeeze().detach().cpu().numpy()
    return keypoints, descriptors

if __name__=='__main__':
    images_list = list(Path("../datas/input/image-matching-challenge-2024/test/church/images/").glob("*.png"))[:10]
    detect_keypoints(images_list)