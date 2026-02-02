#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:52:10 2025

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
import random

# import pandas as pd
# import numpy as np





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

base_path = '/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/train_osm/'

csv_path = os.path.join(base_path, 'Amsterdam_Training_area_bins_images_north_only_DepthBearing.csv')  ### Training file
# csv_path = os.path.join(base_path, 'Amsterdam_New_testing_area_bins_images_DepthBearing.csv')  ### Testing file

#============================================#
#     Adding Noise
#============================================#
# file_name = 'Ams_Train_noisy_1_3'
N1 = '_60_20'
file_name = 'Ams_Train_noisy_1_6'
# augmented_df = add_noise_to_trash_data(csv_path, n_noise=1, bearing_std=3,n_bearing = 1) ## N2, N#
# augmented_df = add_noise_to_trash_data(csv_path, n_noise=1, bearing_std=1.5,n_bearing = (1/2)) ## N1
# augmented_df.to_csv(os.path.join(base_path, file_name + ".csv"), index=False)
# print(augmented_df.head())

#============================================#
#     Removing 20% random views
#============================================#
'''
aug_csv = pd.read_csv(os.path.join(base_path,file_name + ".csv"))
print(f"length if augmented dataset: {len(aug_csv)}")
rem_per = 0.10
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
#==============================================#
'''
# # base_path = '/media/waqar/data3/GNN/Amsterdam/osm_bins/osm_map_bins/'
# base_path = '/media/waqar/data3/GNN_noisy/n_1/subsets_NearObj/train/'
N1 = '_80_10'
in_csv = os.path.join(base_path,file_name + N1 + '.csv')
out_csv = os.path.join(base_path, file_name + N1 + '_ENU.csv')

# # in_csv = os.path.join(base_path,'Amsterdam_New_testing_area_bins_images_DepthBearing.csv')
# # out_csv = os.path.join(base_path,'pano_bins_images_temp_ENU_testing.csv')
cols = Cols(object_id = 'trash_id', cam_lat = 'image_lat' ,cam_lon = 'image_lon',obj_lat = 'trash_lat', obj_lon = 'trash_lon',
                distance_m = 'depth',bearing_deg = 'obj_bearing')
'''

#================================================#
#    Converting to ENU format
#=================================================#
'''
convert_grouped_enu(
    in_csv,
    out_csv,
    cols
)
'''


#==================================================#
#     Traigulation Point Geration
#==================================================#
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
    intersections_csv="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/train_osm/Ams_Train_noisy_1_6_60_20_ENU_hyps_with_noisy.csv",   # your file with columns lon,lat
    tiles_dir="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/train_osm/tiles_osm_train/tiles/",                       # folder with bounds_tile_X_Y.txt and .png
    image_pattern="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/train_osm/tiles_osm_train/tiles//bounds_tile_*.png",
    output_dir="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/train_osm/tiles_osm_train//tiles_with_points/",
    output_flag_csv="/media/waqar/data3/GNN_noisy/n3/subsets_NearObj/train_osm//Ams_Train_noisy_1_6_60_20_ENU_hyps_with_noisy_flags.csv"
)
'''
#==============================================#
#     Graph Genertaion
#==============================================#
'''
# 1) Build graphs for training data
graphs = build_graphs_option1(os.path.join(base_path, file_name + N1 + "_ENU.csv"),
                              os.path.join(base_path, file_name + N1 + "_ENU_hyps_with_noisy_flags.csv"),
                              id_col="trash_id", range_col="depth")  # set None if no ranges

# 1) Build graphs for testing data
# graphs = build_graphs_option1("/media/waqar/data3/GNN/Amsterdam/GNN_v1//pano_bins_images_temp_ENU_testing.csv",
#                               "/media/waqar/data3/GNN/Amsterdam/GNN_v1//hypothesis_testing.csv",
#                               id_col="trash_id", range_col="depth")  # set None if no ranges

torch.save(graphs, os.path.join(base_path, file_name + N1 + "_ENU_hypothesis_with_noisy_flags.pt"))
# print(f"Saved {len(graphs)} graphs to graphs_cache.pt")
'''

#==============================================#
#     Graph Loading
#==============================================#
# '''
graphs = torch.load(os.path.join(base_path, file_name + N1 + "_ENU_hypothesis_with_noisy_flags.pt"))
print(f"Loaded {len(graphs)} graphs from cache")

# records = []

# for g in graphs:
#     # Convert node features to numpy
#     x = g.x.numpy()
#     node_type = g.node_type.numpy()

#     # separate hypothesis nodes
#     hyps_mask = node_type == 1
#     hyps = x[hyps_mask]

#     # osm_flag is the last column
#     osm_flags = hyps[:, -1]

#     # select hypotheses where osm_flag == 1
#     sel = hyps[osm_flags == 1]

#     # collect info
#     for row in sel:
#         tri_E, tri_N, _, _, _, osm_flag = row
#         records.append({
#             "object_id": g.object_id,
#             "tri_E": tri_E,
#             "tri_N": tri_N,
#             "osm_flag": osm_flag
#         })

# # make a DataFrame
# df_osm1 = pd.DataFrame(records)
# print("Total flagged hypotheses:", len(df_osm1))

# 
# '''

#==============================================================#
#    Training and Validation Loop For EdgeAwareVH
#==============================================================#
'''
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Split
N = len(graphs); split = int(0.8 * N)
train_set, val_set = graphs[:split], graphs[split:]
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

# 3) Model/optim

model = HypothesisScorer(in_node_dim=6, in_edge_dim=3, hidden=128, layers=5).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
torch.cuda.empty_cache()
# 4) Train
print('##### Training Started #####')
model_path = base_path + '/models/'
mean_temp = np.inf
for epoch in range(1, 51):
    model.train(); tloss=0.0
    for data in tqdm(train_loader):
        data = data.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(data)                      # [Nh]
        # obj_score = torch.max(logits)
        loss = F.cross_entropy(logits.unsqueeze(0), data.y)  # y is [1] with class idx
        # loss = F.binary_cross_entropy_with_logits(obj_score.unsqueeze(0), (data.y).float())
        loss.backward(); opt.step()
        tloss += loss.item()
        torch.cuda.empty_cache()
    # quick val
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device, non_blocking=True)
            logits = model(data)
            pred = int(torch.argmax(logits))
            total += 1
            if hasattr(data, "y") and pred == int(data.y.item()):
                correct += 1
    print(f"Epoch {epoch:02d} | train loss {tloss/len(train_loader):.4f} | val top1 {correct}/{total}")
#     print(f"Epoch {epoch:02d} | train loss {tloss/len(train_loader):.4f}")
    mean_dis, _, mean_dis_err = evaluate_localization_error(model, val_loader, device)
    if mean_dis_err < mean_temp:
        model_name = model_path + f"EdgeAwareVH_{mean_dis_err:.3}_{N1}.pth"
        torch.save(model, model_name)
        print(f"model saved to the location")
        mean_temp = mean_dis_err
    torch.cuda.empty_cache()


'''
#==============================================================#
#    Training and Validation Loop For EdgeAwareVH
#==============================================================#
'''
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Split
N = len(graphs); split = int(0.8 * N)
train_set, val_set = graphs[:split], graphs[split:]
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

# 3) Model/optim

# model = HypothesisScorer(in_node_dim=6, in_edge_dim=3, hidden=128, layers=5).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model = HypothesisGAT(in_node_dim=6, hidden=64, heads=4, layers=5).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

torch.cuda.empty_cache()
# 4) Train
print('##### Training Started #####')
model_path = base_path + '/models/'
mean_temp = np.inf
for epoch in range(1, 51):
    model.train(); tloss=0.0
    for data in tqdm(train_loader):
        data = data.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(data)                      # [Nh]
        # obj_score = torch.max(logits)
        loss = F.cross_entropy(logits.unsqueeze(0), data.y)  # y is [1] with class idx
        # loss = F.binary_cross_entropy_with_logits(obj_score.unsqueeze(0), (data.y).float())
        loss.backward(); opt.step()
        tloss += loss.item()
        torch.cuda.empty_cache()
    # quick val
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device, non_blocking=True)
            logits = model(data)
            pred = int(torch.argmax(logits))
            total += 1
            if hasattr(data, "y") and pred == int(data.y.item()):
                correct += 1
    print(f"Epoch {epoch:02d} | train loss {tloss/len(train_loader):.4f} | val top1 {correct}/{total}")
#     print(f"Epoch {epoch:02d} | train loss {tloss/len(train_loader):.4f}")
    mean_dis, _, mean_dis_err = evaluate_localization_error(model, val_loader, device)
    if mean_dis_err < mean_temp:
        model_name = model_path + f"GAT_{mean_dis_err:.3}_{N1}.pth"
        torch.save(model, model_name)
        print(f"model saved to the location")
        mean_temp = mean_dis_err
    torch.cuda.empty_cache()
    
    
'''  
#==============================================================#
#    Training and Validation Loop For EdgeAwareVH
#==============================================================#
# '''
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Split
N = len(graphs); split = int(0.8 * N)
train_set, val_set = graphs[:split], graphs[split:]
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

# 3) Model/optim

# model = HypothesisScorer(in_node_dim=6, in_edge_dim=3, hidden=128, layers=5).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# model = HypothesisGAT(in_node_dim=6, hidden=64, heads=4, layers=5).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

model = HypothesisECC(in_node_dim=6, in_edge_dim=3, hidden=16, layers=3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


torch.cuda.empty_cache()
# 4) Train
print('##### Training Started #####')
model_path = base_path + '/models/'
mean_temp = np.inf
for epoch in range(1, 31):
    model.train(); tloss=0.0
    for data in tqdm(train_loader):
        data = data.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(data)                      # [Nh]
        # obj_score = torch.max(logits)
        loss = F.cross_entropy(logits.unsqueeze(0), data.y)  # y is [1] with class idx
        # loss = F.binary_cross_entropy_with_logits(obj_score.unsqueeze(0), (data.y).float())
        loss.backward(); opt.step()
        tloss += loss.item()
        torch.cuda.empty_cache()
    # quick val
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device, non_blocking=True)
            logits = model(data)
            pred = int(torch.argmax(logits))
            total += 1
            if hasattr(data, "y") and pred == int(data.y.item()):
                correct += 1
    print(f"Epoch {epoch:02d} | train loss {tloss/len(train_loader):.4f} | val top1 {correct}/{total}")
#     print(f"Epoch {epoch:02d} | train loss {tloss/len(train_loader):.4f}")
    mean_dis, _, mean_dis_err = evaluate_localization_error(model, val_loader, device)
    if mean_dis_err < mean_temp:
        model_name = model_path + f"ECC_{mean_dis_err:.3}_{N1}.pth"
        torch.save(model, model_name)
        print(f"model saved to the location")
        mean_temp = mean_dis_err
    torch.cuda.empty_cache()


# '''
#=====================================================#
#     Testing The model
#=====================================================#
