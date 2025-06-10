import os
import os.path as osp
import numpy as np
from copy import deepcopy
import pycolmap
pycolmap.logging.minloglevel = 999
import pandas as pd

import pickle
import json
import shutil
import time

from tasks.utils.output_supress import OutputCapture
from reconstruction_utils import *

def Kmean_cluster(df):
    G = build_graph_from_df(df)
    spectral_clusters = spectral_clustering_from_graph(G)
    df['key_1_cluster'] = df['key1'].map(spectral_clusters)
    df['key_2_cluster'] = df['key2'].map(spectral_clusters)

    set_1 = {k for k, v in spectral_clusters.items() if v == 1}
    set_0 = {k for k, v in spectral_clusters.items() if v == 0}

    df_0 = df.loc[(df['key_1_cluster']==0) & (df['key_2_cluster']==0)]
    df_1 = df.loc[(df['key_1_cluster']==1) & (df['key_2_cluster']==1)]
    
    if len(df_0) == 0 or len(df_1) == 0:
        return df, set(), set(), df_0, df_1
    else:
        return df, set_1, set_0, df_0, df_1
    
def run_reconstruction(database_path, image_path, output_path, options):
    with OutputCapture():
        with pycolmap.ostream():
            maps = pycolmap.incremental_mapping(database_path=database_path, image_path=image_path, output_path=output_path, options=options)
            
    print(maps)
    return maps

def Predict_number_of_clusters_KMeans(work_dir_scene):
    df = pd.read_csv(os.path.join(work_dir_scene, "image_pair.csv"))
    clustered_df, set_1, set_0, df_0, df_1 = Kmean_cluster(df)

    if len(set_0) == 0 or len(set_1) == 0:
        print('All the images of one of the 2 clusters are paired with the images of the other cluster - one cluster')
        return 1, {}, {}
    
    if (clustered_df.shape[0] - (df_0.shape[0] + df_1.shape[0])) <= (clustered_df.shape[0] * 0.05):
        return 2, set_1, set_0
    
    return 1, {}, {}

def main_reconstruction(data_dict, dataset, work_dir, work_dir_scene, colmap_mapper_options, n_tries=0, fix_pair=True, 
                        tries_for_second_img=1, images_registered_count={}, verbose=False):
    # Import keypoint distances of matches into colmap for RANSAC 
    images_dir = data_dict[dataset][0].parent
    with open(os.path.join(work_dir, "h_w_exif.json"), "r") as f:
        h_w_exif = json.load(f)
    
    with open(os.path.join(work_dir, "fms.pkl"), "rb") as f:
        fms = pickle.load(f)
    import_into_colmap(work_dir, h_w_exif, fms)

    database_path = os.path.join(work_dir, "colmap.db")
    output_path = os.path.join(work_dir, "colmap_rec_aliked")
    mapper_options = pycolmap.IncrementalPipelineOptions(**colmap_mapper_options)

    db = COLMAPDatabase.connect(database_path)
    cursor = db.execute("SELECT image_id, name from images")
    db_data = cursor.fetchall()
    image_ids = [int(x[0]) for x in db_data]
    names = [str(x[1]) for x in db_data]
    db.close()

    df = pd.read_csv(os.path.join(work_dir_scene, "image_pair.csv"))
    image_scores = {}
    for name in names:
        image_scores[name] = [0, 0]
    for i, row in df.iterrows():
        key1 = row["key1"]
        key2 = row["key2"]
        match_num = row["match_num"]
        image_scores[key1][0] += 1
        image_scores[key1][1] += match_num
        image_scores[key2][0] += 1
        image_scores[key2][1] += match_num

    init_image_name1 = None
    if len(images_registered_count) > 0:
        image_scores_not_used = {im: v for im, v in image_scores.items() if images_registered_count.get(im, 1) == 0}
        importance_scores = sorted(image_scores_not_used.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        if len(importance_scores) > n_tries:
            init_image_name1 = importance_scores[n_tries][0]

    if init_image_name1 is None:
        importance_scores = sorted(image_scores.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        if len(importance_scores) <= n_tries:
            return {}
        init_image_name1 = importance_scores[n_tries][0]
    init_image_id1 = image_ids[names.index(init_image_name1)]
    mapper_options.init_image_id1 = init_image_id1
    print(f"init_image_name1: {init_image_name1} - matched with {image_scores[init_image_name1][0]} images by {image_scores[init_image_name1][1]} kpts !")
    if verbose:
        print(f"all matches: {importance_scores}")

    df_filtered = df.loc[(df["key1"] == init_image_name1) | (df["key2"] == init_image_name1)]
    df_filtered = df_filtered.sort_values(by=["match_num"], ascending=False, ignore_index=True)
    n_tries_for_second_img = min(tries_for_second_img, len(df_filtered))

    for i in range(n_tries_for_second_img):
        if fix_pair:
            best_pair_for_init = df_filtered.iloc[i]
            init_image_name2 = best_pair_for_init["key2"] if best_pair_for_init["key1"] == init_image_name1 else best_pair_for_init["key1"]
            print(f"init_image_name2: {init_image_name2} - matched with {image_scores[init_image_name2][0]} images by {image_scores[init_image_name2][1]} kpts !")
            init_image_id2 = image_ids[names.index(init_image_name2)]
            mapper_options.init_image_id2 = init_image_id2

        if osp.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
    
        maps = run_reconstruction(database_path, images_dir, output_path, mapper_options)
        results = look_for_best_reconstruction_in_map(maps, images_dir, min_model_size=colmap_mapper_options['min_model_size'])
        if len(results) > 0:
            break
    return results

def merge_results(all_results_with_iterative_reconstruction, all_results_with_clustering, favorize_clustering):
    if len(all_results_with_iterative_reconstruction) == 0:
        all_results = all_results_with_clustering
    else:
        if len(all_results_with_clustering) == 0:
            all_results = all_results_with_iterative_reconstruction
        else:
            if favorize_clustering:
                images_register_by_clustering = sum([len(res.keys()) for res in all_results_with_clustering])
                images_register_by_iterations = sum([len(res.keys()) for res in all_results_with_iterative_reconstruction if len(res) > 5])
            else:
                images_register_by_clustering = sum([len(res.keys()) for res in all_results_with_clustering if len(res) > 3])
                images_register_by_iterations = sum([len(res.keys()) for res in all_results_with_iterative_reconstruction if len(res) > 3])
            if images_register_by_clustering >= images_register_by_iterations:
                all_results = all_results_with_clustering
            else:
                all_results = all_results_with_iterative_reconstruction

    return all_results

def iterative_reconstruction(all_images, data_dict, dataset, work_dir, colmap_mapper_options, use_centroids, tries_for_second_img, 
                             rem_only_registered_imgs_having_all_pairs_registered=False, fix_pair=False, verbose=False):
    all_results_with_iterative_reconstruction = []
    
    image_pairs_path = osp.join(work_dir, "image_pair_orig.csv")
    image_pair_df = pd.read_csv(image_pairs_path)
    all_images_names = [osp.basename(im) for im in all_images]
    image_pair_df = image_pair_df.loc[image_pair_df["key1"].isin(all_images_names) & image_pair_df["key2"].isin(all_images_names)]
    image_pair_df.reset_index(drop=True, inplace=True)
    if len(image_pair_df) == 0:
        print('No pair found!')
        return all_results_with_iterative_reconstruction
    groups = get_groups_from_matches(image_pair_df)

    if use_centroids:
        colmap_options_for_clustering = deepcopy(colmap_mapper_options)
        colmap_options_for_clustering["max_num_models"] = 1

        cluster_id = 0
        Atomic_groups = []
        for group in groups:
            if len(group) < 3:
                continue
            scene = f"cluster{cluster_id}"
            # copy all the file in work_dir to work_dir/scene (create if not exists)
            work_dir_scene = os.path.join(work_dir, scene)
            if os.path.exists(work_dir_scene):
                shutil.rmtree(work_dir_scene)
            os.makedirs(work_dir_scene)

            update_output(work_dir, work_dir_scene, group)
            nb_clusters, first_group, second_group = Predict_number_of_clusters_KMeans(work_dir_scene)
            if nb_clusters == 2:
                Atomic_groups.append(first_group)
                Atomic_groups.append(second_group)
            else:
                Atomic_groups.append(group)

        print(f"Number of clusters: {len([group for group in groups if len(group) >= 5])} -> {len(Atomic_groups)}")
    
    if not use_centroids:
        Atomic_groups = deepcopy(groups)
    cluster_id = 0
    for group in Atomic_groups:
        if len(group) < 3:
            continue

        print(f"Cluster {cluster_id}: {len(group)} images")
        scene = f"cluster{cluster_id}"
        # copy all the file in work_dir to work_dir/scene (create if not exists)
        work_dir_scene = os.path.join(work_dir, scene)
        if os.path.exists(work_dir_scene):
            shutil.rmtree(work_dir_scene)
        os.makedirs(work_dir_scene)

        _group = update_output(work_dir, work_dir_scene, group)

        poor_scene = False
        n_tries = 0
        group_subset = deepcopy(_group)
        images_registered_count = {im: 0 for im in group_subset}
        while True:
            if poor_scene:
                rec = all_results_with_iterative_reconstruction[-1]
                print(f"total images: {len(all_images)}, total rec: {len(rec.keys())} - group_subset: {len(group_subset)}")
                if rem_only_registered_imgs_having_all_pairs_registered:
                    group_subset, images_registered_count = get_images_for_next_reconstruction(group_subset, image_pair_df, rec, 
                                                                                               images_registered_count)
                else:
                    group_subset = [osp.basename(im) for im in all_images if im not in rec.keys() and osp.basename(im) in group_subset]
                if len(group_subset) < colmap_mapper_options['min_model_size']:
                    break
                print(f"group_subset: {len(group_subset)}")
                group_subset = update_output(work_dir_scene, work_dir_scene, group_subset)
                print(f"group_subset: {len(group_subset)}")
                if len(group_subset) < colmap_mapper_options['min_model_size']:
                    break
                poor_scene = False
            
            clean_reconstruction(work_dir_scene)
            print(f"Reconstruction number: {len(all_results_with_iterative_reconstruction)}")
            try:
                results = main_reconstruction(data_dict, dataset, work_dir_scene, work_dir_scene, colmap_mapper_options, n_tries, fix_pair, 
                                                tries_for_second_img, images_registered_count, verbose=verbose)
            except:
                results = {}
                print("Error occurred during reconstruction. Trying again...")
            if len(results) != 0:
                all_results_with_iterative_reconstruction.append(results)
                poor_scene = True
                cluster_id += 1
                n_tries = 0
                continue
            if len(group_subset) - n_tries > colmap_mapper_options['min_model_size']:
                n_tries += 1
            else:
                break
        for _ in range(5):
            try:
                shutil.rmtree(work_dir_scene)
                break
            except PermissionError:
                time.sleep(1)
        cluster_id += 1
    return all_results_with_iterative_reconstruction

def cluster_and_reconstruct(data_dict, dataset, work_dir, colmap_mapper_options, thr, tolerance, clustering_threshold, use_centroids, 
                            tries_for_second_img, rem_only_registered_imgs_having_all_pairs_registered, fix_pair_for_iter=False, 
                            fix_pair_for_cluster=False, verbose=False):
    all_images = [str(im) for im in data_dict[dataset]]
    all_results_with_clustering = []
    all_results_with_iterative_reconstruction = []

    cache_output(work_dir)
    successful_reconstructions = False
    favorize_clustering = False
    use_clustering = os.path.exists(os.path.join(work_dir, "embeddings.pkl"))

    if use_clustering:
        # Run the clustering algorithm
        print("Running clustering algorithm")
        embeddings = pickle.load(open(os.path.join(work_dir, "embeddings.pkl"), "rb"))
        scene_groups, outliers, clustering_score = get_scenes(embeddings, all_images, thr)
        if len(scene_groups) != 0:
            print(f"Found {len(scene_groups)} clusters {len(outliers)} outliers")
            successful_reconstructions = True
            for i, (cluster_id, cluster_images) in enumerate(scene_groups.items()):
                print(f"Cluster {i}: {len(cluster_images)} images")
                scene = f"cluster{cluster_id}"
                # copy all the file in work_dir to work_dir/scene (create if not exists)
                work_dir_scene = os.path.join(work_dir, scene)
                if os.path.exists(work_dir_scene):
                    shutil.rmtree(work_dir_scene)
                os.makedirs(work_dir_scene)
                _cluster_images = update_output(work_dir, work_dir_scene, cluster_images)
                print(f"Cluster {i}: {len(_cluster_images)} images")
                n_tries = 0
                cached_results = []
                while True:
                    clean_reconstruction(work_dir)
                    clean_reconstruction(work_dir_scene)
                    if clustering_score > clustering_threshold:
                        results = main_reconstruction(data_dict, dataset, work_dir_scene, work_dir_scene, colmap_mapper_options, n_tries, 
                                                      fix_pair=fix_pair_for_cluster, tries_for_second_img=tries_for_second_img, 
                                                      verbose=verbose)
                    else:
                        results = main_reconstruction(data_dict, dataset, work_dir, work_dir_scene, colmap_mapper_options, n_tries, 
                                                      fix_pair=fix_pair_for_cluster, tries_for_second_img=tries_for_second_img, 
                                                      verbose=verbose)
                    if len(results) != 0 and len(results) > int(len(_cluster_images) * tolerance):
                        all_results_with_clustering.append((results, _cluster_images))
                        break
                    if len(results) != 0:
                        cached_results.append(results)
                    n_tries += 1
                    if n_tries >= 3:
                        if len(cached_results) > 0:
                            best_results = sorted(cached_results, key=lambda x: len(list(x.keys())), reverse=True)[0]
                            all_results_with_clustering.append((best_results, _cluster_images))
                        break
                if len([im for im in all_results_with_clustering[-1][0] if osp.basename(im) in _cluster_images]) < int(len(_cluster_images) * 0.9):
                    successful_reconstructions = False
                shutil.rmtree(work_dir_scene)

            all_results_with_clustering = post_process(all_results_with_clustering, embeddings)
            # If there are still images that are not reconstructed, try to reconstruct
            left_images_names = {osp.basename(im) for im in all_images} - {osp.basename(im) for res in all_results_with_clustering for im in res.keys()}
            print(f'There are {len(left_images_names)} images that are not reconstructed')
            if len(left_images_names) > 5:
                print(f'Try to reconstruct the left images ..')
                left_images = [im for im in all_images if osp.basename(im) in left_images_names]
                extra_results = iterative_reconstruction(
                    left_images, data_dict, dataset, work_dir, colmap_mapper_options, use_centroids=False, 
                    tries_for_second_img=tries_for_second_img, fix_pair=fix_pair_for_cluster, verbose=verbose)
                if len(extra_results) > 0:
                    registered_images = sum([len(res.keys()) for res in extra_results])
                    print(f"Found {len(extra_results)} extra reconstructions with {registered_images} images")
                    all_results_with_clustering.extend(extra_results)

            favorize_clustering = clustering_score > clustering_threshold

    if not successful_reconstructions:
        if use_clustering:
            print("Failed to reconstruct all scenes with clustering. Trying without clustering")
        all_results_with_iterative_reconstruction = iterative_reconstruction(
            all_images, data_dict, dataset, work_dir, colmap_mapper_options, use_centroids, tries_for_second_img, 
            rem_only_registered_imgs_having_all_pairs_registered, fix_pair=fix_pair_for_iter, verbose=verbose)
        
        if rem_only_registered_imgs_having_all_pairs_registered:
            embeddings = pickle.load(open(os.path.join(work_dir, "embeddings.pkl"), "rb"))
            all_results_with_iterative_reconstruction = post_process_for_iter(all_results_with_iterative_reconstruction, embeddings)

    all_results = merge_results(all_results_with_iterative_reconstruction, all_results_with_clustering, favorize_clustering)
    return all_results

def reconstruction(data_dict, dataset, work_dir, colmap_mapper_options, thr_config):
    using_gt_cluster = isinstance(dataset, tuple)
    if using_gt_cluster:
        dataset, scene = dataset
        all_results, _ = main_reconstruction(data_dict[dataset], scene, work_dir, colmap_mapper_options, verbose=thr_config['verbose'])
        all_results = [all_results]
        all_images = [str(im) for im in data_dict[dataset][scene]]
    else:
        all_results = cluster_and_reconstruct(data_dict, dataset, work_dir, colmap_mapper_options, **thr_config)
        all_images = [str(im) for im in data_dict[dataset]]

    # Create Submission
    submission = {
        "image_id": [],
        "dataset": [],
        "scene": [],
        "image": [],
        "rotation_matrix": [],
        "translation_vector": []
    }
    registered_images = []

    for scene_idx, results in enumerate(all_results):
        if using_gt_cluster:
            predicted_scene = scene
        else:
            predicted_scene = f"cluster{scene_idx}"
        print(f"Registered: {dataset} / {predicted_scene} -> {len(results)} images")
        if using_gt_cluster:
            print(f"Total: {dataset} / {predicted_scene} -> {len(data_dict[dataset][scene])} images")
        else:
            print(f"Total: {dataset} / {predicted_scene} -> {len(data_dict[dataset])} images")

        for image in results:
            if image in all_images:
                R = results[image]["R"].reshape(-1)
                T = results[image]["t"].reshape(-1)

            image_name = osp.basename(image)
            submission["image_id"].append(f"{dataset}_{image_name}_public")
            submission["dataset"].append(dataset)
            submission["scene"].append(predicted_scene)
            submission["image"].append(image_name)
            submission["rotation_matrix"].append(arr_to_str(R))
            submission["translation_vector"].append(arr_to_str(T))

            registered_images.append(image)

    none_to_str = lambda n: ';'.join(['nan'] * n)
    for image in list(set(all_images) - set(registered_images)):
        R = none_to_str(9)
        T = none_to_str(3)

        image_name = osp.basename(image)
        submission["image_id"].append(f"{dataset}_{image_name}_public")
        submission["dataset"].append(dataset)
        submission["scene"].append("outliers")
        submission["image"].append(image_name)
        submission["rotation_matrix"].append(R)
        submission["translation_vector"].append(T)

    assert len(set(submission["image"])) == len(all_images)

    submission_df = pd.DataFrame.from_dict(submission)
    submission_path = os.path.join(work_dir, "submission.csv")
    print(f"Save to {submission_path}")
    submission_df.to_csv(submission_path, index=False)
    return submission_path