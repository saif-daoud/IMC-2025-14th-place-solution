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

def embed_images(
    paths: list[Path],
    model_name: str,
    device: torch.device = torch.device("cpu"),
) -> T:
    """Computes image embeddings.
    
    Returns a tensor of shape [len(filenames), output_dim]
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    embeddings = []
    
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Global descriptors"):
        image = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
        
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs) # last_hidden_state and pooled
            
            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=-1, p=2)
            
        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list"""
    return list(itertools.combinations(range(len(lst)), 2))    


def get_image_pairs(
    paths: list[Path],
    model_name: str,
    similarity_threshold: float = 0.6,
    tolerance: int = 1000,
    min_matches: int = 20,
    exhaustive_if_less: int = 20,
    p: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> list[tuple[int, int]]:
    """Obtains pairs of similar images"""
    if len(paths) <= exhaustive_if_less:
        pairs = get_pairs_exhaustive(paths)
        distances = np.zeros((len(paths), len(paths)))
        return pairs, distances

    matches = []
    
    # Embed images and compute distances for filtering
    embeddings = embed_images(paths, model_name, device=device)
    distances = torch.cdist(embeddings, embeddings, p=p)
    
    # Remove pairs above similarity threshold (if enough)
    mask = distances <= similarity_threshold
    image_indices = np.arange(len(paths))
    
    for current_image_index in range(len(paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row]
        
        # We don't have enough matches below the threshold, we pick most similar ones
        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index])[:min_matches]
            
        for other_image_index in indices_to_match:
            # Skip an image matching itself
            if other_image_index == current_image_index:
                continue
            
            # We need to check if we are below a certain distance tolerance 
            # since for images that don't have enough matches, we picked
            # the most similar ones (which could all still be very different 
            # to the image we are analyzing)
            if distances[current_image_index, other_image_index] < tolerance:
                # Add the pair in a sorted manner to avoid redundancy
                matches.append(tuple(sorted((current_image_index, other_image_index.item()))))

    pairs = sorted(list(set(matches)))
    distances = distances.numpy()
    return pairs, distances


def task_get_image_pair_DINO(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    image_paths = params["data_dict"]
    print(f"image_num = {len(image_paths)}")
    input_dir_root = params["input_dir_root"]
    params["pair_matching_args"]["model_name"] = os.path.join(input_dir_root, params["pair_matching_args"]["model_name"])
    pairs, distances = get_image_pairs(
                image_paths,
                **params["pair_matching_args"],
                device=params["device"],
            )
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
        res["sim"].append(distances[p1][p2])
        res["match_num"].append(0)
    res_df = pd.DataFrame.from_dict(res)

    work_dir = params["work_dir"]
    output_path = work_dir / params["output"]
    res_df.to_csv(output_path, index=False)
    print(f"save -> {output_path}")


if __name__=='__main__':
    images_list = list(Path("../datas/input/image-matching-challenge-2024/test/church/images/").glob("*.png"))[:10]
    index_pairs = get_image_pairs(images_list, "../datas/input/dinov2", exhaustive_if_less=0)
    print(index_pairs)