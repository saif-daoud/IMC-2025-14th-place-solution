from typing import Any
import itertools
import pandas as pd
from collections import defaultdict, deque


def get_pairs_exhaustive(lst: list[Any]) -> list[tuple[int, int]]:
    """Obtains all possible index pairs of a list"""
    return list(itertools.combinations(range(len(lst)), 2))

def get_connected_components(image_pairs_path: str) -> list[str]:
    image_pair_df = pd.read_csv(image_pairs_path)
    pairs = list(zip(image_pair_df["key1"], image_pair_df["key2"]))

    # Build adjacency list
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)

    # Find connected components using BFS
    visited = set()
    groups = []

    for node in adj:
        if node not in visited:
            group = set()
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                group.add(current)
                queue.extend(adj[current] - visited)
            groups.append(group)
    groups = [list(group) for group in groups]
    return groups

def task_get_image_pair_exhaustive_within_component(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    image_paths = params["data_dict"]
    print(f"image_num = {len(image_paths)}")

    work_dir = params["work_dir"]
    input_path = work_dir / params["input"]["image_pair"]
    components = get_connected_components(input_path)
    pairs_list = []
    for component in components:
        if len(component) < 3:
            continue
        pairs_list.append((component, get_pairs_exhaustive(component)))

    print(f"pair_num = {sum([len(pairs) for _, pairs in pairs_list])}")
    
    input_path_for_rotation = work_dir / params["input"]["image_pair_rotation"]
    image_pair_rot_df = pd.read_csv(input_path_for_rotation)

    res = {
        "key1": [],
        "key2": [],
        "dir1": [],
        "dir2": [],
        "sim": [],
        "match_num": []
    }
    for component, pairs in pairs_list:
        for pair in pairs:
            p1, p2 = pair
            name1 = component[p1]
            name2 = component[p2]
            
            image_pair_rot_df_filtered1 = image_pair_rot_df.loc[
                (image_pair_rot_df["key1"] == name1) & (image_pair_rot_df["key2"] == name2)
            ]
            image_pair_rot_df_filtered2 = image_pair_rot_df.loc[
                (image_pair_rot_df["key1"] == name2) & (image_pair_rot_df["key2"] == name1)
            ]
            if len(image_pair_rot_df_filtered1) > 0:
                dir1 = image_pair_rot_df_filtered1["dir1"].values[0]
                dir2 = image_pair_rot_df_filtered1["dir2"].values[0]
                sim = image_pair_rot_df_filtered1["sim"].values[0]
            elif len(image_pair_rot_df_filtered2) > 0:
                name1, name2 = name2, name1
                dir1 = image_pair_rot_df_filtered2["dir1"].values[0]
                dir2 = image_pair_rot_df_filtered2["dir2"].values[0]
                sim = image_pair_rot_df_filtered2["sim"].values[0]
            else:
                continue
            res["key1"].append(name1)
            res["key2"].append(name2)
            res["dir1"].append(dir1)
            res["dir2"].append(dir2)
            res["sim"].append(sim)
            res["match_num"].append(0)
    res_df = pd.DataFrame.from_dict(res)

    work_dir = params["work_dir"]
    output_path = work_dir / params["output"]
    res_df.to_csv(output_path, index=False)
    print(f"save -> {output_path}")