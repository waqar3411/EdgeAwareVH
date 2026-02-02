#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 18:52:34 2025

@author: waqar
"""

import sys
sys.path.append('/home/waqar/Downloads/GNN//')

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from GNN_utils import *
from Models import *
from Evaluate import *
import torch
import torch.nn.functional as F

import torch.nn.functional as F
from torch_geometric.loader import DataLoader


@dataclass
class Cols:
    object_id: str = "object_id"
    cam_lat: str = "image_lat"
    cam_lon: str = "image_lon"
    cam_alt: Optional[str] = None          # optional
    obj_lat: str = "object_lat"
    obj_lon: str = "object_lon"
    obj_alt: Optional[str] = None          # optional
    bearing_deg: Optional[str] = "bearing_deg"  # optional
    distance_m: Optional[str] = "distance_m"    # optional

N = 'n3'
base_path = '/media/waqar/data3/GNN_noisy/' + N + '/subsets_NearObj/test_osm/'

# csv_path = os.path.join(base_path, 'Ams_mapillary_images_near_bins_15m_Test_DepthBearing.csv')  ### Training file
csv_path = os.path.join(base_path, 'Amsterdam_New_testing_area_bins_images_DepthBearing.csv')  ### Testing file

#============================================#
#     Adding Noise
#============================================#
# file_name = 'Ams_map_Test_noisy'
# file_name = 'Ams_Test_noisy'
# file_name = 'Ams_test_noisy_1_3'
file_name = 'kitti_inter_cleaned'
# N1 = '_80_10'
N1 = '_kitti'

# file_name = 'Ams_Test_noisy'
# augmented_df = add_noise_to_trash_data(csv_path, n_noise=1, bearing_std=6,n_bearing = 2) ## N2, N#
# augmented_df = add_noise_to_trash_data(csv_path, n_noise=1, bearing_std=(1.5),n_bearing = (1/2)) ## N1
# augmented_df.to_csv(os.path.join(base_path, file_name + ".csv"), index=False)
# print(augmented_df.head())

#============================================#
#     Removing 20% random views
#============================================#
'''
# f_temp = 'Amsterdam_New_testing_area_bins_images_DepthBearing'
aug_csv = pd.read_csv(os.path.join(base_path,file_name + ".csv"))
print(f"length if augmented dataset: {len(aug_csv)}")
rem_per = 0.20
df_dropped = aug_csv.drop(aug_csv.sample(frac=rem_per, random_state=42).index)
print(f"length of {rem_per} dataset: {len(df_dropped)}")
df_dropped.to_csv(os.path.join(base_path, file_name + '_80.csv'), index = False)
'''

#============================================#
#    Adding 10% noise (CNN False Positives)
# ============================================#
'''
df = pd.read_csv(os.path.join(base_path, file_name + '_80.csv'))  # your original dataset

# Generate random subset with new depth & bearing
new_rows, n_added = select_and_randomize(df, group_col="trash_id", frac=0.1)

# --- flag original rows ---
df["is_augmented"] = 0

# --- combine old + new ---
df_aug = pd.concat([df, new_rows], ignore_index=True)

print(f"Total rows after augmentation: {len(df_aug)} (added {n_added})")

# --- optionally save ---
df_aug.to_csv(os.path.join(base_path, file_name + "_80_10.csv"), index=False)
print(f"Saved augmented dataset with size: {len(df_aug)}")
'''

#==============================================#
#    Generating ENU file
# ==============================================#
'''
# # base_path = '/media/waqar/data3/GNN/Amsterdam/osm_bins/osm_map_bins/'
# base_path = '/media/waqar/data3/GNN_noisy/n_1/subsets_NearObj/train/'
# N1 = '_80_10'
in_csv = os.path.join(base_path,file_name + N1 + '.csv')
out_csv = os.path.join(base_path, file_name + N1 + '_ENU.csv')

# # in_csv = os.path.join(base_path,'Amsterdam_New_testing_area_bins_images_DepthBearing.csv')
# # out_csv = os.path.join(base_path,'pano_bins_images_temp_ENU_testing.csv')
cols = Cols(object_id = 'trash_id', cam_lat = 'image_lat' ,cam_lon = 'image_lon',obj_lat = 'trash_lat', obj_lon = 'trash_lon',
                distance_m = 'depth',bearing_deg = 'obj_bearing')
'''

#================================================#
#    Converting to ENU format
# =================================================#
'''
convert_grouped_enu(
    in_csv,
    out_csv,
    cols
)
'''


#==================================================#
#     Traigulation Point Geration
# ==================================================#
'''
binary_triangulation_grouped(
    csv_in=out_csv,
    csv_out_pairs=os.path.join(base_path, file_name + N1 + "_ENU_tri.csv"),
    object_id_col="trash_id",
    residual_thresh=2.0
)
'''


#=================================================#
#     Hypothesis Point Creation
# =================================================#
'''
out_csv = os.path.join(base_path, 'kitti_inter_cleaned_ENU.csv')
df = pd.read_csv(out_csv)
if 'trash_id' not in df.columns:
    raise ValueError(f"Missing object id column: {args.object_id_col}")
need = ["cam_E","cam_N","bearing_sin","bearing_cos"]
for c in need:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

all_rows = []
for oid, g in df.groupby(df['trash_id'], sort=False):
#     print(oid)
    rows = generate_hypotheses_for_group(
        g,
        None,
        0,
        0,
        None
    )
    for r in rows:
#         print(r)
        r["trash_id"] = oid
#         print(r)
    all_rows.extend(rows)

out_df = pd.DataFrame(all_rows,
                      columns=["trash_id","i","j","tri_E","tri_N","residual","src","jitter_k","d_to_gt",
                              "tri_lat", "tri_lon"])
hyp_csv = os.path.join(base_path, file_name + N1 + '_ENU_hypothesis.csv')
out_df.to_csv(hyp_csv, index=False)
'''

#===============================================#
#    Inject Noisy Intersections
#===============================================#
'''
merged_df = inject_noisy_intersections(
    enu_csv = os.path.join(base_path, file_name + N1 + '_ENU.csv'),
    hyps_csv = os.path.join(base_path, file_name + N1 + '_ENU_hypothesis.csv'),
    output_csv = os.path.join(base_path, file_name + N1 + "_ENU_hyps_with_noisy.csv"),
    neighbor_radius=25,
    cross_frac=0.5
)
'''

#===============================================#
#    osmnx intersection flags
#===============================================#
'''

# plot_intersections_on_tiles(
#     intersections_csv=os.path.join(base_path, "Ams_Train_noisy_1_3_80_10_ENU_hyps_with_noisy.csv"),   # your file with columns lon,lat
#     tiles_dir=os.path.join(base_path, "/tiles_osm_train/tiles/"),                    # folder with bounds_tile_X_Y.txt and .png
#     image_pattern=os.path.join(base_path, "/tiles/bounds_tile_*.png"),
#     output_dir= "/media/waqar/data3/GNN_noisy/n1/subsets_NearObj/train_osm/tiles_osm_train/tiles_with_points/",
#     output_flag_csv="/media/waqar/data3/GNN_noisy/n1/subsets_NearObj/train_osm//Ams_Train_noisy_1_3_80_10_ENU_hyps_with_noisy_flags.csv"
# )

plot_intersections_on_tiles(
    intersections_csv="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/test_osm/kitti_inter_cleaned_ENU_hypothesis.csv",   # your file with columns lon,lat
    tiles_dir="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/test_osm/tiles/",                       # folder with bounds_tile_X_Y.txt and .png
    image_pattern="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/test_osm/tiles//bounds_tile_*.png",
    output_dir="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/test_osm/tiles_with_points/",
    output_flag_csv="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/test_osm/kitti_inter_cleaned_kitti_ENU_hypothesis_flags.csv"
)
'''
#==============================================#
#     Graph Genertaion
# ==============================================#
'''
# 1) Build graphs for training data
# graphs = build_graphs_option1(os.path.join(base_path, file_name + N1 + "_ENU.csv"),
                              # os.path.join(base_path, file_name + N1 + "_ENU_hyps_with_noisy_flags.csv"),
                              # id_col="trash_id", range_col="depth")  # set None if no ranges
                              
graphs = build_graphs_option1(os.path.join(base_path, file_name + "_ENU.csv"),
                           os.path.join(base_path, "kitti_inter_cleaned_kitti_ENU_hypothesis_flags.csv"),
                           id_col="trash_id", range_col="depth")  # set None if no ranges

# 1) Build graphs for testing data
# graphs = build_graphs_option1("/media/waqar/data3/GNN/Amsterdam/GNN_v1//pano_bins_images_temp_ENU_testing.csv",
                              # "/media/waqar/data3/GNN/Amsterdam/GNN_v1//hypothesis_testing.csv",
                              # id_col="trash_id", range_col="depth")  # set None if no ranges

torch.save(graphs, os.path.join(base_path, file_name + N1 + "_ENU_hypothesis_with_noisy_flags.pt"))
print(f"Saved {len(graphs)} graphs to graphs_cache.pt")
'''

#==============================================#
#     Graph Loading
#==============================================#
'''
# graphs = torch.load(os.path.join(base_path, file_name + N1 + "_ENU_hypothesis_with_noisy.pt"))
# graphs = torch.load(os.path.join(base_path, "kitti.pt"))
# print(f"Loaded {len(graphs)} graphs from cache")


graphs = torch.load("/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/test_osm/kitti_inter_cleaned_kitti_ENU_hypothesis_with_noisy_flags.pt")

records = []

for g in graphs:
    # Convert node features to numpy
    x = g.x.numpy()
    node_type = g.node_type.numpy()

    # separate hypothesis nodes
    hyps_mask = node_type == 1
    hyps = x[hyps_mask]

    # osm_flag is the last column
    osm_flags = hyps[:, -1]

    # select hypotheses where osm_flag == 1
    sel = hyps[osm_flags == 1]

    # collect info
    for row in sel:
        tri_E, tri_N, _, _, _, osm_flag = row
        records.append({
            "object_id": g.object_id,
            "tri_E": tri_E,
            "tri_N": tri_N,
            "osm_flag": osm_flag
        })

# make a DataFrame
df_osm1 = pd.DataFrame(records)
print("Total flagged hypotheses:", len(df_osm1))

'''
#==============================================#
#    Testing The model
# ==============================================#
# '''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# load Model
# file_name_test = 'Ams_Test_noisy_1_3'
model_path = '/media/waqar/data3/GNN_noisy/'+ N + '/subsets_NearObj/train_osm/models/'
# model_name = 'EdgeAwareVH_0.6535381716766967_N0.pth'
# model_name = 'EdgeAwareVH_2.114723600197723_N1.pth'
# model_name = 'EdgeAwareVH_2.182091381799566_N2.pth'
# model_name = 'EdgeAwareVH_2.826739480511795_N3.pth'
# model_name = 'EdgeAwareVH_2.111315143424143_N_1.pth'
# model_name = 'EdgeAwareVH_2.088729384711406_N1.pth' # osm flags
# model_name = 'GAT_4.69__60_20.pth'
# model_name = 'ECC_0.63__60_20.pth'
model_name = 'EdgeAwareVH_0.612__60_20.pth'
model = torch.load(os.path.join(model_path,model_name))
model.to(device)
# print(model)

# load test graph
graphs = torch.load(os.path.join(base_path, "kitti_inter_cleaned_kitti_ENU_hypothesis_with_noisy_flags.pt"))
# graphs = torch.load(os.path.join(base_path,  "depths_Ams_camFirst.pt"))
# graphs = torch.load("/media/waqar/data3/GNN/Amsterdam/osm_bins/kitti/cleaned_by_image_coords_ENU_hypothesis.pt")
# print(f"Loaded {len(graphs)} graphs name: {file_name}")

# enu_csv = base_path + file_name + N1 + '_ENU.csv'
enu_csv = base_path + file_name + '_ENU.csv'
# enu_csv = '/media/waqar/data3/GNN/Amsterdam/osm_bins/kitti/cleaned_by_image_coords_ENU.csv'
val_loader   = DataLoader(graphs,   batch_size=1, shuffle=False)
preds, hyp_conf, max_conf = predict_objects_(model, val_loader,
                        enu_csv, device)

# import pandas as pd
df_preds = pd.DataFrame(preds)
print(df_preds.head())

df_conf = pd.DataFrame(hyp_conf)
print(df_conf.head())

max_conf = pd.DataFrame(max_conf)
print(max_conf.head())

# Save for later analysis
# df_preds.to_csv(os.path.join(base_path,"osm_map_common_bins_testing_predictions_vs_gt_noisy.csv"), index=False)
df_preds.to_csv(os.path.join(base_path, file_name + "_pred_vs_gt.csv"), index=False)
df_conf.to_csv(os.path.join(base_path, file_name + "_conf.csv"), index=False)
max_conf.to_csv(os.path.join(base_path, file_name + "_max_conf.csv"), index=False)

merged_df = merge_close_predictions(
    pred_csv=df_preds,
    out_csv=os.path.join(base_path, file_name + "_merged_predictions_"+ N +"_eps_10.csv"),
    eps_meters=10.0  # merge points within 1 meter
)
'''

'''

# kitti_df = predict_NMS(model, val_loader,device)
# kitti_df.to_csv('/media/waqar/data3/GNN/Amsterdam/osm_bins/kitti/pred_ENU_score.csv', index = False)
# '''

#===========================================
# Precision and recall Calculation
# ===========================================
'''
# ---------- 1️⃣  Load CSVs ----------
# file_name = 'kitti_inter_cleaned'
N = 'n3'
# N = 'MRF_no_sim'
base_path = '/media/waqar/data3/GNN_noisy/' + N + '/subsets_NearObj/test_osm/'
# base_path = '/media/waqar/data3/New_data_Ams_DCC/Amsterdam/New_testing_out/MRF_with_sim/'
# base_path = '/media/waqar/data3/New_data_Ams_DCC/Amsterdam/New_testing_out/MRF_No_sim/'
gt_csv = '/media/waqar/data3/GNN_noisy/' + "Amsterdam_20km_bins_selected_test_grouped.csv"
# gt_csv = '/media/waqar/data3/GNN_noisy/' + "Amsterdam_New_testing_area_bins_images_DepthBearing.csv"
pred_csv = base_path + file_name + "_merged_predictions_" + N + "_eps_10_ECC.csv"
# pred_csv = base_path + "0.3_perfect1_pairs.csv"
# pred_csv = base_path + "0_perfect1_pairs.csv"
df_gt = pd.read_csv(gt_csv)
df_pred = pd.read_csv(pred_csv)
print(f"length of detected objects: {len(df_pred)}")

# ---------- 2️⃣  Group GT by trash_id ----------
if "trash_id" in df_gt.columns:
    df_gt_grouped = (
        df_gt.groupby("trash_id")[["trash_lat", "trash_lon"]]
        .median()
        .reset_index()
    )
else:
    raise ValueError("Ground truth file must contain a 'trash_id' column")

print(f"Unique GT objects: {len(df_gt_grouped)}")

# ---------- 3️⃣  Haversine distance in meters ----------
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000.0
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
#     return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# ---------- 4️⃣  Match predicted ↔ GT points ----------
matches = []
used_pred = set()
used_gt = set()
DIST_THRESH = 10  # meters

for i, gt_row in df_gt_grouped.iterrows():
    best_j, best_d = None, np.inf
    for j, pred_row in df_pred.iterrows():
        d = haversine(gt_row["trash_lat"], gt_row["trash_lon"],
                      pred_row["merged_lat"], pred_row["merged_lon"])
        if d < best_d:
            best_d, best_j = d, j
    # if best_d <= DIST_THRESH:
        # used_gt.add(gt_row["trash_id"])

# for i, gt_row in df_gt_grouped.iterrows():
#     best_j, best_d = None, np.inf
#     for j, pred_row in df_pred.iterrows():
#         if j in used_pred:
#             continue
#         d = haversine(gt_row["trash_lat"], gt_row["trash_lon"],
#                       pred_row["merged_lat"], pred_row["merged_lon"])
#         if d < best_d:
#             best_d, best_j = d, j

    if best_d <= DIST_THRESH:
        matches.append({
            "trash_id": gt_row["trash_id"],
            "gt_lat": gt_row["trash_lat"],
            "gt_lon": gt_row["trash_lon"],
            "pred_lat": df_pred.loc[best_j, "merged_lat"],
            "pred_lon": df_pred.loc[best_j, "merged_lon"],
            "distance_m": best_d,
            "flag": 1
        })
        used_pred.add(best_j)
        used_gt.add(gt_row["trash_id"])

# ---------- 5️⃣  Add False Positives ----------
for j, pred_row in df_pred.iterrows():
    if j not in used_pred:
        matches.append({
            "trash_id": np.nan,
            "gt_lat": np.nan,
            "gt_lon": np.nan,
            "pred_lat": pred_row["merged_lat"],
            "pred_lon": pred_row["merged_lon"],
            "distance_m": np.nan,
            "flag": 0
        })

# ---------- 6️⃣  Save output ----------
df_out = pd.DataFrame(matches)
out_csv = base_path + file_name + "_pred_vs_gt_" + str(DIST_THRESH) + "m_eval_" + N + "_ECC.csv"
df_out.to_csv(out_csv, index=False)
print(f"[OK] Saved: {out_csv}")

# ---------- 7️⃣  Compute Precision & Recall ----------
TP = np.sum(df_out["flag"] >= 1)
# TP = 124
FP = np.sum(df_out["flag"] == 0)
# FP = 47
FN = len(df_gt_grouped) - len(used_gt)
# FN = len(df_gt_grouped) - len(used_gt)

# ---------- 8️⃣  Extract and Save False Negatives ----------
false_negatives = df_gt_grouped[~df_gt_grouped["trash_id"].isin(used_gt)].copy()
false_negatives["flag"] = -1  # optional marker for FN

fn_csv = base_path + file_name + "_false_negatives_" + str(DIST_THRESH) + "_" + N + ".csv"
false_negatives.to_csv(fn_csv, index=False)

print(f"[OK] Saved False Negatives → {fn_csv}")
print(f"Total False Negatives saved: {len(false_negatives)}")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision = {precision:.3f}")
print(f"Recall    = {recall:.3f}")
'''