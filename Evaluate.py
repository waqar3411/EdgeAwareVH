#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:40:54 2025

@author: waqar
"""

import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.append('/home/waqar/Downloads/GNN//')

from GNN_utils import *

def predict_objects_(model, data_loader, enu_csv, device,
                    id_col="trash_id", topk=1, mode="median"):
    """
    Predict object positions using top-k hypotheses per graph.

    mode: 'mean' → weighted mean ENU
          'median' → median ENU (robust to outliers)
    """
    df = pd.read_csv(enu_csv)
    results = []
    hyp_rows = []

    model.eval()
    max_rows = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data)                       # [Nh]
            probs  = F.softmax(logits, dim=0)          # [Nh]
            
            best_h = int(torch.argmax(probs))
            best_logit = float(logits[best_h])
            best_prob  = float(probs[best_h])

            # --- top-k indices ---
            Nh = len(probs)
            
            
            
            
            k = min(topk, Nh)
            topk_idx = torch.topk(probs, k).indices.cpu()

            # --- get ENU coordinates for top-k ---
            hyps_EN = data.hyps_EN[topk_idx].cpu().numpy()  # shape [k,2]
            
            best_E = float(data.hyps_EN[best_h][0])
            best_N = float(data.hyps_EN[best_h][1])
            
            weights = probs[topk_idx].cpu().numpy()

            # --- weighted mean or median ---
            if mode == "mean":
                pred_EN = np.average(hyps_EN, axis=0, weights=weights)
            elif mode == "median":
                pred_EN = np.median(hyps_EN, axis=0)
            else:
                raise ValueError("mode must be 'mean' or 'median'")
                
            # --- Confidence metrics ---
            conf_sum  = float(weights.sum())       # total belief mass of top-k
            conf_max  = float(weights.max())       # highest prob among top-k
            entropy   = -np.sum(weights * np.log(weights + 1e-9))
            conf_entropy = 1.0 if k == 1 else 1.0 - entropy / np.log(k)

            # --- get object origin ---
            oid = int(data.object_id)
            g = df[df[id_col] == oid]
            if g.empty:
                print(f"[WARN] No origin found for object {oid}, skipping.")
                continue

            olat = float(g["origin_lat"].median())
            olon = float(g["origin_lon"].median())
            oh   = float(g["origin_h"].median()) if "origin_h" in g.columns else 0.0

            # --- ENU → WGS84 ---
            pred_lat, pred_lon, _ = enu_to_latlon(
                float(pred_EN[0]), float(pred_EN[1]), 0.0, olat, olon, oh
            )

            # --- ground truth if available ---
            if hasattr(data, "gt_xy"):
                gt_EN = data.gt_xy.cpu().numpy()
                gt_lat, gt_lon, _ = enu_to_latlon(
                    float(gt_EN[0]), float(gt_EN[1]), 0.0, olat, olon, oh
                )
                err = float(np.linalg.norm(pred_EN - gt_EN))
            else:
                gt_lat, gt_lon, err = np.nan, np.nan, np.nan

            results.append({
                "object_id": int(oid),
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "error_m": err,
                "confidence_sum": conf_sum,
                "confidence_max": conf_max,
                "confidence_entropy": conf_entropy,
            })
            # --------------------------------------------------------
            # Save per-hypothesis confidence for CSV
            # --------------------------------------------------------
            for h in range(Nh):
                E = float(data.hyps_EN[h][0])
                N = float(data.hyps_EN[h][1])
            
                # Convert to WGS84 coordinates
                hyp_lat, hyp_lon, _ = enu_to_latlon(E, N, 0.0, olat, olon, oh)
                hyp_rows.append({
                    "object_id": int(data.object_id),
                    "hyp_index": h,
                    "logit": float(logits[h].cpu()),
                    "softmax_prob": float(probs[h].cpu()),
                    "E": float(data.hyps_EN[h][0]),
                    "N": float(data.hyps_EN[h][1]),
                    "lat": hyp_lat,
                    "lon": hyp_lon,
                })
            # --------------------------------------------------------
            
            best_lat, best_lon, _ = enu_to_latlon(best_E, best_N, 0.0, olat, olon, oh)
            max_rows.append({
                    "object_id": oid,
                    "best_hyp_index": best_h,
                    "logit": best_logit,
                    "softmax_prob": best_prob,
                    "lat": best_lat,
                    "lon": best_lon,
                })

    print(f"[OK] Predictions completed for {len(results)} objects (top-{topk}, mode={mode}).")
    return results, hyp_rows, max_rows

def evaluate_localization_error(model, val_loader, device, topk_frac=0.1, dist_thresh=2.0):
    model.eval()
    errors = []
    correct = 0
    total = 0
    all_dists, all_scores = [], []
    all_errors = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            logits = model(data)                     # [Nh]
            probs  = F.softmax(logits, dim=0)        # [Nh]
            
            # --- Select top-K high-confidence hypotheses ---
            Nh = len(probs)
            k_top = max(1, int(topk_frac * Nh))
            topk_idx = torch.topk(probs, k_top).indices
            
            k = int(torch.argmax(probs))             # predicted hyp index
            pred_EN = data.hyps_EN[k].cpu().numpy()  # [E,N]
            
            if hasattr(data, "gt_xy"):
                gt_EN = data.gt_xy.cpu().numpy()     # [E,N]
                err = float(np.linalg.norm(pred_EN - gt_EN))
                errors.append(err)

            # --- Compute distances for all top-K ---
            hyps_EN = data.hyps_EN[topk_idx].cpu().numpy()
            dists = np.linalg.norm(hyps_EN - gt_EN, axis=1)
            all_errors.extend(dists.tolist())
            # Top-1 correctness check
            if hasattr(data, "y"):
                if k == int(data.y.item()):
                    correct += 1
                total += 1

    if len(all_errors) == 0:
        return None, None, None

    mean_d = np.mean(all_errors)
    med_d  = np.median(all_errors)
    # Summarize
    if errors:
        mean_err = np.mean(errors)
        med_err  = np.median(errors)
        print(f"Val: mean error = {mean_err:.2f} m | mean dis = {mean_d:.3f} | median error = {med_err:.2f} m")
    if total > 0:
        print(f"Val: top-1 accuracy = {correct}/{total} ({100*correct/total:.1f}%)")
        
    return mean_err, med_err, mean_d




def predict_with_gt(model, val_loader, enu_csv,device, id_col="object_id"):
    """
    Run inference on val_loader and return list of dicts with:
      object_id, pred_lat, pred_lon, gt_lat, gt_lon, error_m
    """
    import pandas as pd
    df = pd.read_csv(enu_csv)   # to fetch origin lat/lon per object
#     print(df)

    results = []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            logits = model(data)                     
            probs  = F.softmax(logits, dim=0)
            k = int(torch.argmax(probs))              # predicted hyp index
            pred_EN = data.hyps_EN[k].cpu().numpy()   # [E,N]
            print(pred_EN)

            # --- convert predicted ENU back to lat/lon ---
            oid = int(data.object_id)
#             print(oid)
            g = df[df[id_col] == oid]
#             print(g)
            olat = float(g["origin_lat"].median())
            olon = float(g["origin_lon"].median())
#             print(f"[DEBUG] oid={oid}, origin=({olat}, {olon}), pred_EN={pred_EN}")

            oh   = float(g["origin_h"].median()) if "origin_h" in g.columns else 0.0

            # ENU -> lat/lon conversion
            pred_lat, pred_lon, _ = enu_to_latlon(
                float(pred_EN[0]), float(pred_EN[1]), 0.0,
                olat, olon, oh
            )

            # --- ground truth (lat/lon) ---
            if hasattr(data, "gt_xy"):
                gt_EN = data.gt_xy.cpu().numpy()
                gt_lat, gt_lon, _ = enu_to_latlon(
                    float(gt_EN[0]), float(gt_EN[1]), 0.0,
                    olat, olon, oh
                )
                err = float(np.linalg.norm(pred_EN - gt_EN))
            else:
                gt_lat, gt_lon, err = np.nan, np.nan, np.nan

            results.append({
                "object_id": int(oid),
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "error_m": err
            })

    return results


def predict_objects(model, data_loader, enu_csv,device, id_col="trash_id"):
    """
    Run inference on a loader (val/test) and return a list of dicts with:
      object_id, pred_lat, pred_lon, [gt_lat, gt_lon, error_m if available]

    Works for both GT (val) and non-GT (test) data.
    """

    # --- Load ENU origin info for each object ---
    df = pd.read_csv(enu_csv)
    results = []

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data)                      # [num_hypotheses]
            probs  = F.softmax(logits, dim=0)
            k = int(torch.argmax(probs))              # predicted hypothesis index

            # --- Get predicted ENU coordinates ---
            pred_EN = data.hyps_EN[k].cpu().numpy()   # shape: [2] → [E, N]
            confidence = float(probs[k].cpu().item())

            # --- Fetch ENU origin for this object ---
            oid = int(data.object_id)
            g = df[df[id_col] == oid]
            if g.empty:
                print(f"[WARN] No origin found for object {oid}, skipping.")
                continue

            olat = float(g["origin_lat"].median())
            olon = float(g["origin_lon"].median())
            oh   = float(g["origin_h"].median()) if "origin_h" in g.columns else 0.0

            # --- Convert predicted ENU → WGS84 lat/lon ---
            pred_lat, pred_lon, _ = enu_to_latlon(
                float(pred_EN[0]), float(pred_EN[1]), 0.0, olat, olon, oh
            )

            # --- Optional: ground truth if available ---
            if hasattr(data, "gt_xy"):
                gt_EN = data.gt_xy.cpu().numpy()
                gt_lat, gt_lon, _ = enu_to_latlon(
                    float(gt_EN[0]), float(gt_EN[1]), 0.0, olat, olon, oh
                )
                err = float(np.linalg.norm(pred_EN - gt_EN))
            else:
                gt_lat, gt_lon, err = np.nan, np.nan, np.nan

            # --- Store result ---
            results.append({
                "object_id": int(oid),
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "error_m": err,
                "confidence": confidence
            })

    print(f"[OK] Predictions completed for {len(results)} objects.")
    return results





def predict_NMS(model, val_loader,device):
    results = []
    model.eval()
    
    # --- Inference loop ---
    with torch.no_grad():
        for data in tqdm(val_loader):   # DataLoader containing your graph objects
            data = data.to(device)
            logits = model(data)                     # [Nh] raw scores for hypothesis nodes
            probs  = torch.sigmoid(logits).cpu()     # convert to probability in [0,1]
            hypsEN = data.hyps_EN.cpu().numpy()      # [Nh,2] coordinates (E,N)
            oid    = data.object_id
    
            for (E, N, s) in zip(hypsEN[:,0], hypsEN[:,1], probs.numpy()):
                results.append({
                    "object_id": oid.item(),
                    "tri_E": E,
                    "tri_N": N,
                    "EdgeAwareVH_score": s
                })
    
    df_results = pd.DataFrame(results)
    # print(df_results.head())
    return df_results

