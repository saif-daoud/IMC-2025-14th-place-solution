import os
import pandas as pd

def task_get_transparent_pair(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    input_df = pd.read_csv(os.path.join(work_dir, params["input"]))

    key_list = list(set(input_df["key1"].values.tolist() + input_df["key2"].values.tolist()))
    remove_too_many_matches = {}
    registerd_counter = {}
    for k in key_list:
        # match_nums = input_df.query("key1==@k or key2==@k")["match_num"].values
        # if (match_nums>1000).sum() > 5:
        #     remove_too_many_matches[k] = True
        # else:
        #     remove_too_many_matches[k] = False
        remove_too_many_matches[k] = False
        
        # initialize
        registerd_counter[k] = 0


    pair_list = []
    match_num_list = []
    for i, row in input_df.iterrows():
        key1 = row["key1"]
        key2 = row["key2"]
        match_num = row["match_num"]

        if (remove_too_many_matches[key1] or remove_too_many_matches[key2]) and match_num > 1000:
            match_num = 0
        
        pair_list.append((key1, key2))
        match_num_list.append(match_num)
    
    sorted_pairs = sorted(zip(pair_list, match_num_list), key=lambda x:x[1], reverse=True)
    sorted_pair_list, sorted_match_num_list = zip(*sorted_pairs)
    best_pair_list = []
    for pair, m in zip(sorted_pair_list, sorted_match_num_list):
        p1, p2 = pair
        if registerd_counter[p1]>=params["k"] or registerd_counter[p2]>=params["k"]:
            continue

        best_pair_list.append(pair)
        registerd_counter[p1] += 1
        registerd_counter[p2] += 1


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
        if pair not in best_pair_list and rev_pair not in best_pair_list:
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