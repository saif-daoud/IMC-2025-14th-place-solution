import os
import os.path as osp
import pickle

def task_copy_dino_embeddings(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    emb_dir = params["input"]["embedding_dir"]
    emb_name = params["input"]["embeddings_pkl"]
    work_dir = params["work_dir"]
    scene = osp.basename(work_dir)

    with open(os.path.join(emb_dir, scene, emb_name), "rb") as f:
        emb = pickle.load(f)

    output_path = os.path.join(work_dir, params["output"])
    with open(output_path, "wb") as f:
        pickle.dump(emb, f)