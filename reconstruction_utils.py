import os
import os.path as osp
import numpy as np
import pandas as pd
from copy import deepcopy

from collections import defaultdict, deque
from PIL import ExifTags
from tqdm import tqdm
import networkx as nx
import warnings
import sqlite3
import pickle
import shutil
import h5py
import json

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import hdbscan
import pycolmap

def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2

def array_to_blob(array):
    return array.tostring()

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=0, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)
        if image_id1 > image_id2:
            matches = matches[:,::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches, F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)
        if image_id1 > image_id2:
            matches = matches[:,::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

        
def get_focal(height, width, exif):
    max_size = max(height, width)
    focal_found, focal = False, None
    if exif is not None:
        focal_35mm = None
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
        if focal_35mm is not None:
            focal_found = True
            focal = focal_35mm / 35. * max_size
            print(f"Focal found: {focal}")
    if focal is None:
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
    return focal_found, focal


def create_camera(db, height, width, exif, camera_model):
    focal_found, focal = get_focal(height, width, exif)
    if camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'radial':
        model = 3 # radial
        param_arr = np.array([focal, width / 2, height / 2, 0., 0.])
    elif camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
    return db.add_camera(model, width, height, param_arr, prior_focal_length=int(focal_found))

def add_keypoints(db, feature_dir, h_w_exif, camera_model, single_camera=False):
    keypoint_f = h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r')
    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]
        if camera_id is None or not single_camera:
            height = h_w_exif[filename]['h']
            width = h_w_exif[filename]['w']
            exif = h_w_exif[filename]['exif']
            camera_id = create_camera(db, height, width, exif, camera_model)
        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id
        db.add_keypoints(image_id, keypoints)
    return fname_to_id

def add_matches_and_fms(db, feature_dir, fname_to_id, fms):
    match_file = h5py.File(os.path.join(feature_dir, 'matches.h5'), 'r')
    added = set()
    for key_1 in match_file.keys():
        group = match_file[key_1]
        for key_2 in group.keys():
            id_1 = fname_to_id[key_1]
            id_2 = fname_to_id[key_2]
            pair_id = (id_1, id_2)
            if pair_id in added:
                warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                continue
            added.add(pair_id)
            matches = group[key_2][()]
            db.add_matches(id_1, id_2, matches)
            db.add_two_view_geometry(id_1, id_2, matches, fms[(key_1, key_2)])

def import_into_colmap(feature_dir, h_w_exif, fms):
    database_path = f"{feature_dir}/colmap.db"
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_dir, h_w_exif, camera_model='simple-radial', single_camera=False)
    add_matches_and_fms(db, feature_dir, fname_to_id, fms)
    db.commit()
    db.close()

def cache_output(work_dir):
    if os.path.exists(os.path.join(work_dir, "h_w_exif_orig.json")):
        # This will be excuted after the first iteration and it uses the results from the first iteration
        shutil.copy2(os.path.join(work_dir, "h_w_exif_orig.json"), os.path.join(work_dir, "h_w_exif.json"))
        shutil.copy2(os.path.join(work_dir, "fms_orig.pkl"), os.path.join(work_dir, "fms.pkl"))
        shutil.copy2(os.path.join(work_dir, "image_pair_orig.csv"), os.path.join(work_dir, "image_pair.csv"))
        shutil.copy2(os.path.join(work_dir, "keypoints_orig.h5"), os.path.join(work_dir, "keypoints.h5"))
        shutil.copy2(os.path.join(work_dir, "matches_orig.h5"), os.path.join(work_dir, "matches.h5"))
        # remove colmap_rec_aliked, colmap.db
        if osp.exists(os.path.join(work_dir, "colmap_rec_aliked")):
            shutil.rmtree(os.path.join(work_dir, "colmap_rec_aliked"))
        if osp.exists(os.path.join(work_dir, "colmap.db")):
            os.remove(os.path.join(work_dir, "colmap.db"))
    else:
        # This will be be excuted only the first time
        shutil.copy2(os.path.join(work_dir, "h_w_exif.json"), os.path.join(work_dir, "h_w_exif_orig.json"))
        shutil.copy2(os.path.join(work_dir, "fms.pkl"), os.path.join(work_dir, "fms_orig.pkl"))
        shutil.copy2(os.path.join(work_dir, "image_pair.csv"), os.path.join(work_dir, "image_pair_orig.csv"))
        shutil.copy2(os.path.join(work_dir, "keypoints.h5"), os.path.join(work_dir, "keypoints_orig.h5"))
        shutil.copy2(os.path.join(work_dir, "matches.h5"), os.path.join(work_dir, "matches_orig.h5"))

def update_output(work_dir, work_dir_scene, images_for_next_reconstruction):
    # remove colmap_rec_aliked, colmap.db
    if osp.exists(os.path.join(work_dir, "colmap_rec_aliked")):
        shutil.rmtree(os.path.join(work_dir, "colmap_rec_aliked"))
    if osp.exists(os.path.join(work_dir, "colmap.db")):
        os.remove(os.path.join(work_dir, "colmap.db"))

    # update image_pair.csv
    df = pd.read_csv(os.path.join(work_dir, "image_pair.csv"))
    df = df.loc[df["key1"].isin(images_for_next_reconstruction) & df["key2"].isin(images_for_next_reconstruction)]
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(work_dir_scene, "image_pair.csv"), index=False)

    refined_images_for_next_reconstruction = list(set(df["key1"].tolist() + df["key2"].tolist()))

    # update h_w_exif.json
    with open(os.path.join(work_dir, "h_w_exif.json"), "r") as f:
        h_w_exif = json.load(f)
    new_h_w_exif = {}
    for k, v in h_w_exif.items():
        if osp.basename(k) in refined_images_for_next_reconstruction:
            new_h_w_exif[k] = v
    with open(os.path.join(work_dir_scene, "h_w_exif.json"), "w") as f:
        json.dump(new_h_w_exif, f, indent=4)
    
    # update fms.pkl
    with open(os.path.join(work_dir, "fms.pkl"), "rb") as f:
        fms = pickle.load(f)
    new_fms = {}
    for k, v in fms.items():
        if osp.basename(k[0]) in refined_images_for_next_reconstruction and osp.basename(k[1]) in refined_images_for_next_reconstruction:
            new_fms[k] = v
    with open(os.path.join(work_dir_scene, "fms.pkl"), "wb") as f:
        pickle.dump(new_fms, f)

    # update keypoints.h5
    keypoints = {}
    with h5py.File(os.path.join(work_dir, "keypoints.h5"), mode="r") as f_keypoints:
        for key in f_keypoints.keys():
            if key in refined_images_for_next_reconstruction:
                keypoints[key] = f_keypoints[key][...]

    with h5py.File(os.path.join(work_dir_scene, "keypoints.h5"), mode="w") as f_keypoints:
        for k, v in keypoints.items():
            f_keypoints[k] = v

    # update matches.h5
    matches = {}
    with h5py.File(os.path.join(work_dir, "matches.h5"), mode="r") as f_matches:
        for key1 in f_matches.keys():
            if key1 in refined_images_for_next_reconstruction:
                matches[key1] = {}
                for key2 in f_matches[key1].keys():
                    if key2 in refined_images_for_next_reconstruction:
                        matches[key1][key2] = f_matches[key1][key2][...]
                if len(matches[key1]) == 0:
                    matches.pop(key1)
    with h5py.File(os.path.join(work_dir_scene, "matches.h5"), mode="w") as f_matches:
        for key1 in matches.keys():
            for key2 in matches[key1].keys():
                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=matches[key1][key2])

    return refined_images_for_next_reconstruction

def fuse_score(sil_score, confidence_score, combo_mode='harmonic'):
    if combo_mode == 'harmonic':
        # it is basically the F1 score
        if (sil_score + confidence_score) == 0:
            score = 0
        else:
            score = 2 * sil_score * confidence_score / (sil_score + confidence_score)
    elif combo_mode == 'geometric':
        score = (sil_score * confidence_score) ** 0.5
    elif combo_mode == 'arithmetic':
        # to be avoided, since if one of the mAA or clusterness score is zero is not zero
        score = (sil_score + confidence_score) * 0.5
    elif combo_mode == 'silhouette_score':
        score = sil_score
    elif combo_mode == 'confidence':
        score = confidence_score
    return score

def post_process(all_results, embeddings):
    # Post-process the results to assign images that are registered to more than one model to the model with the highest average
    # similarity score
    using_groups = isinstance(all_results[0], tuple)
    if using_groups:
        all_results_post = [[im for im in res[0]] for res in all_results]
    else:
        all_results_post = [[im for im in res] for res in all_results]

    n_rec = len(all_results)
    for i in range(n_rec):
        for j in range(i+1, n_rec):
            rec1_keys = all_results_post[i]
            rec2_keys = all_results_post[j]
            overlap = set(rec1_keys).intersection(set(rec2_keys))
            if using_groups:
                grp1 = all_results[i][1]
                grp2 = all_results[j][1]
            if len(overlap) > 0:
                # Get all names of images that do not overlap for each model
                if using_groups:
                    names1 = [osp.basename(im) for im in rec1_keys if im not in overlap and osp.basename(im) in grp1]
                    names2 = [osp.basename(im) for im in rec2_keys if im not in overlap and osp.basename(im) in grp2]
                else:
                    names1 = [osp.basename(im) for im in rec1_keys if im not in overlap]
                    names2 = [osp.basename(im) for im in rec2_keys if im not in overlap]

                # Get the embeddings of the images that do not overlap
                if len(names1) > 2 and len(names2) > 2:
                    emb1 = np.concatenate([embeddings[name] for name in names1], axis=0)
                    emb2 = np.concatenate([embeddings[name] for name in names2], axis=0)
                    for im in overlap:
                        im_name = osp.basename(im)
                        emb = embeddings[im_name]
                        # get the L2 distance between the embedding of the image and the embeddings of the images that do not overlap
                        dist1 = np.linalg.norm(emb - emb1, axis=1)
                        dist2 = np.linalg.norm(emb - emb2, axis=1)
                        # Assign the image to the model with the lowest distance
                        if np.mean(dist1) < np.mean(dist2):
                            # remove the image from all_results_post[j]
                            all_results_post[j] = [res for res in all_results_post[j] if res != im]
                        else:
                            # remove the image from all_results_post[i]
                            all_results_post[i] = [res for res in all_results_post[i] if res != im]
                elif len(names1) > 2:
                    # remove the image from all_results_post[j]
                    all_results_post[j] = [image for image in all_results_post[j] if image not in overlap]
                elif len(names2) > 2:
                    # remove the image from all_results_post[i]
                    all_results_post[i] = [image for image in all_results_post[i] if image not in overlap]
                else:
                    # remove the image from all_results_post[j]
                    all_results_post[j] = [image for image in all_results_post[j] if image not in overlap]

    if using_groups:
        all_results_postprocessed = [{k: v for k, v in res[0].items() if k in all_results_post[i]} for i, res in enumerate(all_results)]
    else:
        all_results_postprocessed = [{k: v for k, v in res.items() if k in all_results_post[i]} for i, res in enumerate(all_results)]

    return all_results_postprocessed

def post_process_for_iter(all_results, embeddings):
    # Post-process the results to assign images that are registered to more than one model to the model with the highest average
    # similarity score
    all_results_post = [[im for im in res] for res in all_results]

    n_rec = len(all_results)
    for i in range(n_rec):
        for j in range(i+1, n_rec):
            rec1_keys = all_results_post[i]
            rec2_keys = all_results_post[j]
            overlap = set(rec1_keys).intersection(set(rec2_keys))
            if len(overlap) > 0:
                # Get all names of images that do not overlap for each model
                names1 = [osp.basename(im) for im in rec1_keys if im not in overlap]
                names2 = [osp.basename(im) for im in rec2_keys if im not in overlap]
                # if len(overlap) > len(names2):
                #     # remove the image from all_results_post[j]
                #     all_results_post[j] = [image for image in all_results_post[j] if image not in overlap]
                # Get the embeddings of the images that do not overlap
                if len(names1) > 2 and len(names2) > 2:
                    n = min(len(names1), len(names2))
                    emb1 = np.concatenate([embeddings[name] for name in names1], axis=0)
                    emb2 = np.concatenate([embeddings[name] for name in names2], axis=0)
                    for im in overlap:
                        im_name = osp.basename(im)
                        emb = embeddings[im_name]
                        # get the L2 distance between the embedding of the image and the embeddings of the images that do not overlap
                        dist1 = np.linalg.norm(emb - emb1, axis=1)
                        dist2 = np.linalg.norm(emb - emb2, axis=1)
                        # get the lowest n distances
                        dist1 = np.sort(dist1)[:n]
                        dist2 = np.sort(dist2)[:n]
                        # Assign the image to the model with the lowest distance
                        if np.mean(dist1) < np.mean(dist2):
                            # remove the image from all_results_post[j]
                            all_results_post[j] = [res for res in all_results_post[j] if res != im]
                        else:
                            # remove the image from all_results_post[i]
                            all_results_post[i] = [res for res in all_results_post[i] if res != im]
                elif len(names1) > 2:
                    # remove the image from all_results_post[j]
                    all_results_post[j] = [image for image in all_results_post[j] if image not in overlap]
                elif len(names2) > 2:
                    # remove the image from all_results_post[i]
                    all_results_post[i] = [image for image in all_results_post[i] if image not in overlap]
                else:
                    # remove the image from all_results_post[j]
                    all_results_post[j] = [image for image in all_results_post[j] if image not in overlap]

    all_results_postprocessed = [{k: v for k, v in res.items() if k in all_results_post[i]} for i, res in enumerate(all_results)]
    return all_results_postprocessed

def look_for_best_reconstruction_in_map(maps, images_dir, min_model_size):
    images_registered = 0
    best_idx = None
    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx1
            except Exception:
                continue

    # Parse the reconstruction object to get the rotation matrix and translation vector
    # obtained for each image in the reconstruction
    results = {}
    camid_im_map = {}
    if best_idx is not None:
        for k, im in maps[best_idx].images.items():
            key = os.path.join(images_dir, im.name)
            results[key] = {}
            results[key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
            results[key]["t"] = deepcopy(np.array(im.cam_from_world.translation))
            camid_im_map[im.camera_id] = im.name
    if len(results) < min_model_size:
        results = {}

    return results

def cluster_with_outliers(features, min_cluster_size=4):
    np.random.seed(42)
    features = StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=min(30.0, features.shape[0] - 1), random_state=42)
    tsne_proj = tsne.fit_transform(features)
    np.random.seed(42)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
    labels = clusterer.fit_predict(tsne_proj)
    if np.mean(labels) == -1:
        labels = labels * 0
    return (labels, clusterer.probabilities_, tsne_proj)

def get_scenes(embeddings, all_images, thr):
    images_list = [osp.basename(im) for im in all_images]
    assert set(images_list) == set(embeddings.keys())
    images, embeddings_np = [], []
    for im, emb in embeddings.items():
        images.append(im)
        embeddings_np.append(emb)
    embeddings_np = np.concatenate(embeddings_np, axis=0)
    labels, confidence_scores, tsne_proj = cluster_with_outliers(embeddings_np)
    
    average_confidence = np.mean(confidence_scores[labels != -1])
    # There is more than one cluster
    if max(labels) > 0:
        # Only use non-outliers for silhouette score
        mask = labels != -1
        sil_score = silhouette_score(tsne_proj[mask], labels[mask])
        clustering_score = fuse_score(sil_score, average_confidence)
        print("Average clustering confidence (non-outliers):", round(average_confidence, 4))
        print("Silhouette Score (non-outliers):", round(sil_score, 4))
        print("Clustreing score (F1): ", round(clustering_score, 4))

        if clustering_score > thr:
            scene_groups = {}
            outliers = []

            for im, label in zip(images, labels):
                if label == -1:
                    outliers.append(im)
                else:
                    scene_groups.setdefault(label, []).append(im)
            return scene_groups, outliers, clustering_score
    return {}, [], 0

def get_groups_from_matches(match_df):
    pairs = list(zip(match_df["key1"], match_df["key2"]))

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
    return groups

def build_graph_from_df(df, match_threshold=0):
    G = nx.Graph()
    for _, row in df.iterrows():
        k1, k2, w = row['key1'], row['key2'], row['match_num']
        if w >= match_threshold:
            G.add_edge(k1, k2, weight=w)
    return G

def spectral_clustering_from_graph(G):
    # Adjacency matrix
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

    # Spectral Clustering
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=0)
    labels = clustering.fit_predict(A)

    return dict(zip(nodes, labels))

def return_centroid_from_df(df_i):
    all_keys = pd.concat([df_i["key1"], df_i["key2"]])
    key_counts = all_keys.value_counts()

    # Get the key with the highest count
    most_common_key = key_counts.idxmax()
    most_common_count = key_counts.max()
    print(f"Most common key: {most_common_key} (appears {most_common_count} times)")
        
    df_filtered = df_i.loc[(df_i["key1"] == most_common_key) | (df_i["key2"] == most_common_key)]
    best_pair_for_init = df_filtered.sort_values(by=["match_num"], ascending=False).iloc[0]
    best_matched_key = best_pair_for_init["key2"] if best_pair_for_init["key1"] == most_common_key else best_pair_for_init["key1"]

    return most_common_key, best_matched_key

def clean_reconstruction(work_dir):
    # remove colmap_rec_aliked, colmap.db
    if osp.exists(os.path.join(work_dir, "colmap_rec_aliked")):
        shutil.rmtree(os.path.join(work_dir, "colmap_rec_aliked"))
    if osp.exists(os.path.join(work_dir, "colmap.db")):
        os.remove(os.path.join(work_dir, "colmap.db"))

def get_images_for_next_reconstruction(images, image_pair_df, reconstruction, images_registered_count):
    # get names of images in reconstruction
    images_in_reconstruction = [osp.basename(im) for im in reconstruction.keys()]

    images_registered_count = {im: c + 1 if im in images_in_reconstruction else c for im, c in images_registered_count.items()}
    # Filter image_pair_df to only include images
    image_pair_df_filtered = image_pair_df.loc[image_pair_df["key1"].isin(images) & image_pair_df["key2"].isin(images)]
    
    images_to_be_rem = [im for im, c in images_registered_count.items() if c == 2]
    for im in images_in_reconstruction:
        im_pairs = image_pair_df_filtered.loc[(image_pair_df_filtered["key1"] == im) | (image_pair_df_filtered["key2"] == im)]
        im_pairs = set(im_pairs['key1'].tolist() + im_pairs['key2'].tolist()) - set([im])
        # Check if all of the im_pairs are registered in reconstruction
        flag = False
        for im_pair in im_pairs:
            if im_pair not in images_in_reconstruction:
                flag = True
                break
        if not flag:
            # if all of the im_pairs are registered in reconstruction, remove the image
            images_to_be_rem.append(im)
    images_for_next_reconstruction = [im for im in images if im not in images_to_be_rem]
    return images_for_next_reconstruction, images_registered_count