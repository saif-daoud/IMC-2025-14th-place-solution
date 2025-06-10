import os
from tqdm import tqdm
import pickle
import torch
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import kornia as K

def task_get_dino_embeddings(params):
    """Computes image embeddings and save it to a file.
    """
    if params["pdb"]:
        import pdb;pdb.set_trace()

    image_paths = params["data_dict"]
    device = params["device"]
    model_name = os.path.join(params["input_dir_root"], params["model_name"])
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    embeddings = {}
    for image_path in tqdm(image_paths, desc="Global descriptors for clustering"):
        image = K.io.load_image(image_path, K.io.ImageLoadType.RGB32, device=device)[None, ...]
        
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs) # last_hidden_state and pooled
            
            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=-1, p=2)
        
        embeddings[os.path.basename(image_path)] = embedding.detach().cpu().numpy()

    work_dir = params["work_dir"]
    output_path = os.path.join(work_dir, params["output"])
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

if __name__=='__main__':
    from pathlib import Path
    images_list = list(Path(r"data\input\train\imc2023_haiper").glob("*.png"))
    input_dir_root = os.path.join("data", "input")
    output_dir = r"data\output\exp4\debug\feature_outputs\imc2023_haiper_cluster0"
    params = {
        'pdb': False,
        'data_dict': images_list,
        'device': "cuda:0",
        'model_name': "dinov2-pytorch-large-v1",
        'input_dir_root': input_dir_root,
        'work_dir': output_dir,
        'output': "embeddings.pkl"
    }
    task_get_dino_embeddings(params)