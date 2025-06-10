import os
import pandas as pd

def task_get_image_pair_kNN(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    work_dir = params["work_dir"]
    input_df = pd.read_csv(os.path.join(work_dir, params["input"]))

    key_list = list(set(input_df["key1"].values.tolist() + input_df["key2"].values.tolist()))
    data_list = {}
    for k in key_list:
        data_list[k] = {
            "ref_key": [],
            "match_num": [],
        }

    for i, row in input_df.iterrows():
        key1 = row["key1"]
        key2 = row["key2"]
        match_num = row["match_num"]

        data_list[key1]["ref_key"].append(key2)
        data_list[key1]["match_num"].append(match_num)
        data_list[key2]["ref_key"].append(key1)
        data_list[key2]["match_num"].append(match_num)
    
    # kNN
    kNN_pair_list = []
    for key, value in data_list.items():
        sorted_pairs = sorted(zip(value["match_num"], value["ref_key"]), key=lambda x:x[0], reverse=True)
        sorted_match_num_list, sorted_ref_key_list = zip(*sorted_pairs)
        nearest_keys = sorted_ref_key_list[:params["k"]]
        for nearest_key in nearest_keys:
            kNN_pair_list.append((key, nearest_key))
    
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
        if pair not in kNN_pair_list and rev_pair not in kNN_pair_list:
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