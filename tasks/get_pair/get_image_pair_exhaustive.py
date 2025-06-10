import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import torch
from torch import Tensor as T
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import kornia as K
from typing import Any
import itertools
import pandas as pd


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list"""
    return list(itertools.combinations(range(len(lst)), 2))    


def task_get_image_pair_exhaustive(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    image_paths = params["data_dict"]
    print(f"image_num = {len(image_paths)}")
    pairs = get_pairs_exhaustive(image_paths)
    print(f"pair_num = {len(pairs)}")
    
    res = {
        "key1": [],
        "key2": [],
        "dir1": [],
        "dir2": [],
        "sim": [],
        "match_num": []
    }
    for pair in pairs:
        p1, p2 = pair
        res["key1"].append(image_paths[p1].name)
        res["key2"].append(image_paths[p2].name)
        res["dir1"].append(0)
        res["dir2"].append(0)
        res["sim"].append(0.0)
        res["match_num"].append(0)
    res_df = pd.DataFrame.from_dict(res)

    work_dir = params["work_dir"]
    output_path = work_dir / params["output"]
    res_df.to_csv(output_path, index=False)
    print(f"save -> {output_path}")