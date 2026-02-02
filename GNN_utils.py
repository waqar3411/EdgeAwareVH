#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:24:52 2025

@author: waqar
"""


import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import torch
from torch_geometric.data import Data
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import cKDTree
from math import radians, cos
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np



def add_noise_to_trash_data(csv_in, n_noise=1, bearing_std=3,n_bearing = 1):
    """
    Create noisy augmented rows per trash_id.
    
    Parameters:
        csv_in (str): path to input CSV file
        n_noise (int): number of noisy samples to generate per original row
        bearing_std (float): std deviation for bearing noise (in degrees)
    """
    # Read input
    df = pd.read_csv(csv_in)
    
    # Store noisy samples
    noisy_rows = []
    
    for _, row in df.iterrows():
        for _ in range(n_noise):
            # Add Gaussian noise
            noisy_bearing = row['obj_bearing'] + np.random.normal(0, bearing_std)
            noisy_depth = row['depth'] + np.random.normal(0, (n_bearing * np.sqrt(row['depth'])))
            
            # Clip physically invalid values
            if noisy_depth < 0:
                noisy_depth = abs(noisy_depth)
            noisy_bearing = noisy_bearing % 360  # wrap around degrees
            
            # Create new row
            new_row = row.copy()
            new_row['obj_bearing'] = noisy_bearing
            new_row['depth'] = noisy_depth
            
            # Optional: mark as noisy
            new_row['is_noisy'] = 1
            
            noisy_rows.append(new_row)
    
    # Combine with original
    df['is_noisy'] = 0
    aug_df = pd.concat([df, pd.DataFrame(noisy_rows)], ignore_index=True)
    
    # Sort by trash_id for clarity
    aug_df = aug_df.sort_values(['trash_id', 'image_id']).reset_index(drop=True)
    
    return aug_df

def select_and_randomize(df, group_col="trash_id",
                         frac=0.1,
                         depth_range=(3, 30),
                         bearing_range=(0, 360),
                         random_state=42):
    """
    Select 10% of rows per bin (group_col), assign random depth and obj_bearing,
    and flag them as augmented.

    Args:
        df : pandas DataFrame
        group_col : str — column name to group by (e.g. 'trash_id')
        frac : float — fraction of rows per group to select
        depth_range : tuple(float,float) — (min,max) range for depth (m)
        bearing_range : tuple(float,float) — (min,max) range for bearing (degrees)
        random_state : int — random seed

    Returns:
        new_rows : DataFrame of selected & modified rows (with flag)
        count_added : int — number of new rows
    """
    np.random.seed(random_state)

    # --- randomly select 10% of rows per bin ---
    sampled_rows = (
        df.groupby(group_col, group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=random_state))
          .reset_index(drop=True)
    )

    # --- randomize depth & bearing ---
    sampled_rows["depth"] = np.random.uniform(depth_range[0], depth_range[1], len(sampled_rows))
    sampled_rows["obj_bearing"] = np.random.uniform(bearing_range[0], bearing_range[1], len(sampled_rows))

    # --- flag augmented rows ---
    sampled_rows["is_augmented"] = 1

    count_added = len(sampled_rows)
    print(f"[OK] Added {count_added} augmented rows ({frac*100:.1f}% per {group_col})")

    return sampled_rows, count_added



# ---- WGS84 helpers (same as before) ----
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)
_WGS84_B = _WGS84_A * (1 - _WGS84_F)
_E2_PRIME = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_B**2

def _deg2rad(d): return d * math.pi / 180.0
def _rad2deg(r): return r * 180.0 / math.pi

def geodetic_to_ecef(lat_deg, lon_deg, h=0.0):
    lat = _deg2rad(lat_deg); lon = _deg2rad(lon_deg)
    sinφ, cosφ = math.sin(lat), math.cos(lat)
    sinλ, cosλ = math.sin(lon), math.cos(lon)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sinφ**2)
    X = (N + h) * cosφ * cosλ
    Y = (N + h) * cosφ * sinλ
    Z = (N * (1 - _WGS84_E2) + h) * sinφ
    return np.array([X, Y, Z], dtype=float)

def _enu_rotation(lat0_deg, lon0_deg):
    lat0 = _deg2rad(lat0_deg); lon0 = _deg2rad(lon0_deg)
    sinφ, cosφ = math.sin(lat0), math.cos(lat0)
    sinλ, cosλ = math.sin(lon0), math.cos(lon0)
    R = np.array([
        [-sinλ,            cosλ,            0.0],
        [-sinφ*cosλ, -sinφ*sinλ,  cosφ],
        [ cosφ*cosλ,  cosφ*sinλ,  sinφ]
    ], dtype=float)
    return R

def geodetic_to_enu(lat_deg, lon_deg, h_m, origin_lat, origin_lon, origin_h=0.0):
    XYZ = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    X0, Y0, Z0 = geodetic_to_ecef(origin_lat, origin_lon, origin_h)
    R = _enu_rotation(origin_lat, origin_lon)
    d = XYZ - np.array([X0, Y0, Z0], dtype=float)
    return R @ d  # (E,N,U)

# ---- Configurable columns ----
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

def choose_origin(df_obj: pd.DataFrame, cols: Cols, mode: str) -> Tuple[float, float, float]:
    """Pick origin (lat,lon,h) for this object's group."""
    if mode == "cam_centroid":
        lat = float(df_obj[cols.cam_lat].astype(float).mean())
        lon = float(df_obj[cols.cam_lon].astype(float).mean())
        if cols.cam_alt and cols.cam_alt in df_obj:
            h = float(df_obj[cols.cam_alt].astype(float).mean())
        else:
            h = 0.0
        return lat, lon, h
    elif mode == "object_gt":
        # use median of object GT (robust w.r.t. duplicates)
        lat = float(df_obj[cols.obj_lat].astype(float).median())
        lon = float(df_obj[cols.obj_lon].astype(float).median())
        if cols.obj_alt and cols.obj_alt in df_obj:
            h = float(df_obj[cols.obj_alt].astype(float).median())
        else:
            h = 0.0
        return lat, lon, h
    elif mode == "first_cam":
        row = df_obj.iloc[0]
        lat = float(row[cols.cam_lat]); lon = float(row[cols.cam_lon])
        if cols.cam_alt and cols.cam_alt in df_obj:
            h = float(row[cols.cam_alt])
        else:
            h = 0.0
        return lat, lon, h
    else:
        raise ValueError("origin_mode must be one of: cam_centroid, object_gt, first_cam")

def convert_grouped_enu(
    in_csv: str,
    out_csv: str,
    cols: Cols,
    origin_mode: str = "first_cam"
):
    df = pd.read_csv(in_csv)
    required = [cols.object_id, cols.cam_lat, cols.cam_lon, cols.obj_lat, cols.obj_lon]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Prepare output columns
    df["origin_lat"] = np.nan
    df["origin_lon"] = np.nan
    df["origin_h"]   = np.nan

    df["cam_E"] = np.nan; df["cam_N"] = np.nan; df["cam_U"] = np.nan
    df["obj_E"] = np.nan; df["obj_N"] = np.nan; df["obj_U"] = np.nan

    # Optional: bearing sin/cos
    if cols.bearing_deg and cols.bearing_deg in df.columns:
        bearing_rad = np.deg2rad(df[cols.bearing_deg].astype(float) % 360.0)
        df["bearing_sin"] = np.sin(bearing_rad)
        df["bearing_cos"] = np.cos(bearing_rad)

    # Convert per object_id group
    for oid, g in tqdm(df.groupby(cols.object_id, sort=False)):
        o_lat, o_lon, o_h = choose_origin(g, cols, origin_mode)
        idx = g.index

        # camera altitudes
        cam_h = g[cols.cam_alt].astype(float).fillna(0.0).to_numpy() if (cols.cam_alt and cols.cam_alt in g) else np.zeros(len(g))
        # object altitudes
        obj_h = g[cols.obj_alt].astype(float).fillna(0.0).to_numpy() if (cols.obj_alt and cols.obj_alt in g) else np.zeros(len(g))

        cam_lat = g[cols.cam_lat].astype(float).to_numpy()
        cam_lon = g[cols.cam_lon].astype(float).to_numpy()
        obj_lat = g[cols.obj_lat].astype(float).to_numpy()
        obj_lon = g[cols.obj_lon].astype(float).to_numpy()

        cam_E = np.zeros(len(g)); cam_N = np.zeros(len(g)); cam_U = np.zeros(len(g))
        obj_E = np.zeros(len(g)); obj_N = np.zeros(len(g)); obj_U = np.zeros(len(g))

        for k in range(len(g)):
            e, n, u = geodetic_to_enu(cam_lat[k], cam_lon[k], cam_h[k], o_lat, o_lon, o_h)
            cam_E[k], cam_N[k], cam_U[k] = e, n, u
            e, n, u = geodetic_to_enu(obj_lat[k], obj_lon[k], obj_h[k], o_lat, o_lon, o_h)
            obj_E[k], obj_N[k], obj_U[k] = e, n, u

        df.loc[idx, "origin_lat"] = o_lat
        df.loc[idx, "origin_lon"] = o_lon
        df.loc[idx, "origin_h"]   = o_h

        df.loc[idx, "cam_E"] = cam_E; df.loc[idx, "cam_N"] = cam_N; df.loc[idx, "cam_U"] = cam_U
        df.loc[idx, "obj_E"] = obj_E; df.loc[idx, "obj_N"] = obj_N; df.loc[idx, "obj_U"] = obj_U

    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote per-object ENU CSV to: {out_csv}")
    print(f"Origin mode: {origin_mode}  | Objects processed: {df[cols.object_id].nunique()}")



# ==========================================================
# --- ENU ↔ ECEF ↔ LatLon conversion utilities (WGS84) ---
# ==========================================================
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)
_WGS84_B = _WGS84_A * (1 - _WGS84_F)
_E2_PRIME = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_B**2


def enu_to_ecef(E, N, U, lat0, lon0, h0=0.0):
    R = _enu_rotation(lat0, lon0)
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, h0)
    return (R.T @ np.array([E, N, U], float)) + np.array([X0, Y0, Z0], float)

def ecef_to_geodetic(X, Y, Z):
    lon = math.atan2(Y, X)
    p = math.hypot(X, Y)
    θ = math.atan2(Z * _WGS84_A, p * _WGS84_B)
    sinθ, cosθ = math.sin(θ), math.cos(θ)
    lat = math.atan2(
        Z + _E2_PRIME * _WGS84_B * sinθ**3,
        p - _WGS84_E2 * _WGS84_A * cosθ**3
    )
    sinφ = math.sin(lat)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sinφ * sinφ)
    h = p / math.cos(lat) - N
    return _rad2deg(lat), _rad2deg(lon), h

def enu_to_latlon(E, N, U, origin_lat, origin_lon, origin_h=0.0):
    X, Y, Z = enu_to_ecef(E, N, U, origin_lat, origin_lon, origin_h)
    return ecef_to_geodetic(X, Y, Z)


def latlon_to_ENU(lat, lon, ref_lat, ref_lon):
    """Approximate local ENU in meters using reference (flat-earth)."""
    dE = (lon - ref_lon) * 111320 * math.cos(math.radians(ref_lat))
    dN = (lat - ref_lat) * 110540
    return dE, dN

def ENU_to_latlon(E, N, ref_lat, ref_lon):
    """Inverse of latlon_to_ENU."""
    lat = ref_lat + (N / 110540)
    lon = ref_lon + (E / (111320 * math.cos(math.radians(ref_lat))))
    return lat, lon

# ==========================================================
# Helper: Haversine distance (meters)
# ==========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# ==========================================================
# --- Intersection-based triangulation ---
# ==========================================================
def closest_point_between_2d_rays(P1, d1, P2, d2, MaxObjectDstFromCam=30):
    """Intersect-style ENU triangulation"""
    E1, N1 = P1[0], P1[1]
    E2, N2 = P2[0], P2[1]
    dE1, dN1 = d1[0], d1[1]
    dE2, dN2 = d2[0], d2[1]

    a1 = dE1; b1 = dE2; c1 = E2 - E1
    a2 = dN1; b2 = dN2; c2 = N2 - N1

    denom = (a2 * b1 - b2 * a1)
    if abs(denom) < 1e-9:
        return np.array([np.nan, np.nan]), np.inf

    y = (a1 * c2 - a2 * c1) / denom
    if abs(a1) > 1e-9:
        x = (b1 * y + c1) / a1
    else:
        x = (b2 * y + c2) / a2

    if (x < 0 or y < 0 or x > MaxObjectDstFromCam or y > MaxObjectDstFromCam):
        return np.array([np.nan, np.nan]), np.inf

    tri_E1 = E1 + dE1 * x
    tri_N1 = N1 + dN1 * x
    tri_E2 = E2 + dE2 * y
    tri_N2 = N2 + dN2 * y

    resid = np.sqrt((tri_E1 - tri_E2)**2 + (tri_N1 - tri_N2)**2)
    tri_E_mid = (tri_E1 + tri_E2) / 2.0
    tri_N_mid = (tri_N1 + tri_N2) / 2.0

    return np.array([tri_E_mid, tri_N_mid]), resid



# ==========================================================
# --- Triangulation grouped per object, auto ENU→LatLon ---
# ==========================================================
def binary_triangulation_grouped(
    csv_in,
    csv_out_pairs,
    object_id_col="trash_id",
    residual_thresh=None,
    max_pairs_per_object=None
):
    """
    Triangulate within each object group using ENU intersections.
    Automatically converts triangulated ENU coordinates to WGS84 lat/lon.
    """

    df = pd.read_csv(csv_in).copy()
    need = {object_id_col, "cam_E", "cam_N", "bearing_sin", "bearing_cos",
            "origin_lat", "origin_lon"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    all_rows = []

    for oid, g in df.groupby(object_id_col, sort=False):
        cams = g[["cam_E", "cam_N"]].to_numpy(float)
        dirs = np.stack([g["bearing_sin"].to_numpy(float),
                         g["bearing_cos"].to_numpy(float)], axis=1)
        or_lat = g["origin_lat"].median()
        or_lon = g["origin_lon"].median()

        idx_within = list(range(len(g)))
        idx_global = list(g.index.to_list())

        pairs = list(itertools.combinations(idx_within, 2))
        if max_pairs_per_object is not None:
            pairs = pairs[:max_pairs_per_object]

        for i_local, j_local in pairs:
            P1, d1 = cams[i_local], dirs[i_local]
            P2, d2 = cams[j_local], dirs[j_local]
            mid, resid = closest_point_between_2d_rays(P1, d1, P2, d2)

            if np.isnan(mid[0]) or np.isnan(mid[1]):
                continue
            if (residual_thresh is not None) and (resid > residual_thresh):
                continue

            tri_E, tri_N = mid
            tri_lat, tri_lon, _ = enu_to_latlon(tri_E, tri_N, 0, or_lat, or_lon)

            all_rows.append({
                "object_id": oid,
                "i": i_local,
                "j": j_local,
                "view_row_i": idx_global[i_local],
                "view_row_j": idx_global[j_local],
                "tri_E": tri_E,
                "tri_N": tri_N,
                "residual": resid,
                "origin_lat": or_lat,
                "origin_lon": or_lon,
                "tri_lat": tri_lat,
                "tri_lon": tri_lon
            })

    out = pd.DataFrame(
        all_rows,
        columns=["object_id","i","j","view_row_i","view_row_j",
                 "tri_E","tri_N","residual","origin_lat","origin_lon",
                 "tri_lat","tri_lon"]
    )
    out.to_csv(csv_out_pairs, index=False)
    print(f"[OK] Wrote triangulated results with lat/lon to {csv_out_pairs} (rows={len(out)})")



def generate_hypotheses_for_group(g: pd.DataFrame,
                                  residual_thresh: None,
                                  jitter_count: int,
                                  jitter_sigma_m: float,
                                  max_pairs: None):
    """
    Generate triangulated hypotheses (ENU) for one object using the new Intersect-style ray intersection.

    Args:
        g : DataFrame for a single object with at least columns:
            cam_E, cam_N, bearing_sin, bearing_cos
        residual_thresh : max allowable residual distance (None = no filtering)
        jitter_count : number of jittered hypotheses to create per valid pair
        jitter_sigma_m : std. dev. (in meters) for jitter
        max_pairs : limit number of image pairs to use (None = all)
    Returns:
        list of dicts containing triangulated hypotheses
    """
    cams = g[["cam_E", "cam_N"]].to_numpy(float)
    dirs = np.stack([
        g["bearing_sin"].to_numpy(float),
        g["bearing_cos"].to_numpy(float)
    ], axis=1)

    have_gt = ("obj_E" in g.columns) and ("obj_N" in g.columns)
    if have_gt:
        gt = np.array([float(g["obj_E"].median()), float(g["obj_N"].median())], dtype=float)

    idx_rows = list(range(len(g)))
    pairs = list(itertools.combinations(idx_rows, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

        
    or_lat = g["origin_lat"].median()
    or_lon = g["origin_lon"].median()
    
    rows_out = []

    for (i, j) in pairs:
        P1, d1 = cams[i], dirs[i]
        P2, d2 = cams[j], dirs[j]

        # --- new Intersect-style function call ---
        mid, resid = closest_point_between_2d_rays(P1, d1, P2, d2)

        # Skip invalid intersections
        if np.isnan(mid[0]) or np.isnan(mid[1]) or np.isinf(resid):
            continue
        if (residual_thresh is not None) and (resid > residual_thresh):
            continue
        
        tri_E, tri_N = mid
        tri_lat, tri_lon, _ = enu_to_latlon(tri_E, tri_N, 0, or_lat, or_lon)
        base_row = {
            "i": i, "j": j,
            "tri_E": float(mid[0]),
            "tri_N": float(mid[1]),
            "residual": float(resid),
            "src": "pair",
            "jitter_k": 0,
            "tri_lat": tri_lat,
            "tri_lon": tri_lon
        }
        
        
        if have_gt:
            base_row["d_to_gt"] = float(np.linalg.norm(mid - gt))
        rows_out.append(base_row)

        # --- Generate jittered versions around each hypothesis ---
        for k in range(1, jitter_count + 1):
            jitter = np.random.normal(scale=jitter_sigma_m, size=2)
            hyp = mid + jitter
            r = {
                "i": i, "j": j,
                "tri_E": float(hyp[0]),
                "tri_N": float(hyp[1]),
                "residual": float(resid),
                "src": "pair",
                "jitter_k": int(k),
                "tri_lat": tri_lat,
                "tri_lon": tri_lon
            }
            if have_gt:
                r["d_to_gt"] = float(np.linalg.norm(hyp - gt))
            rows_out.append(r)

    return rows_out


# build_graphs_option1.py

def _bearing_err_and_dist(camE, camN, bsin, bcos, hypE, hypN):
    vE, vN = hypE - camE, hypN - camN
    dist = float((vE**2 + vN**2) ** 0.5 + 1e-9)
    uE, uN = vE / dist, vN / dist
    dot = float(np.clip(uE * bsin + uN * bcos, -1.0, 1.0))
    berr = float(math.acos(dot))
    return berr, dist

def build_graphs_option1(
    enu_csv: str,         # per-view file (after ENU step)
    hyps_csv: str,        # your hypotheses file (trash_id,i,j,tri_E,tri_N,residual,src,jitter_k,d_to_gt)
    id_col: str = "trash_id",
    range_col: str = "depth",   # None if you don't have per-image distance
    osm_col: str = "osm_flag"
):
    df = pd.read_csv(enu_csv)
    hy = pd.read_csv(hyps_csv)# 1) Build graphs for training data

    need_views = {id_col, "cam_E", "cam_N", "bearing_sin", "bearing_cos"}
    if not need_views.issubset(df.columns):
        raise ValueError(f"ENU CSV missing columns: {need_views - set(df.columns)}")
    need_hy = {id_col, "tri_E", "tri_N"}
    if not need_hy.issubset(hy.columns):
        raise ValueError(f"Hypotheses CSV missing columns: {need_hy - set(hy.columns)}")

    have_range = (range_col is not None) and (range_col in df.columns)
    have_gt_xy = {"obj_E", "obj_N"}.issubset(df.columns)  # optional, for logging
    have_osm = (osm_col in hy.columns)


    data_list = []
    for oid, g in tqdm(df.groupby(id_col, sort=False)):
        # Hypotheses for this object
        H = hy.loc[hy[id_col] == oid]
        if H.empty:
            continue
        hypsEN = H[["tri_E", "tri_N"]].to_numpy(float)  # [Nh,2]

        # ----- node features -----
        camsEN = g[["cam_E", "cam_N"]].to_numpy(float)
        bsin = g["bearing_sin"].to_numpy(float)
        bcos = g["bearing_cos"].to_numpy(float)
        rng = g[range_col].astype(float).to_numpy() if have_range else None
        # osm = g[]

        feats = []
        types = []
        # views first
        for i in range(len(camsEN)):
            if have_range and np.isfinite(rng[i]) and rng[i] > 0:
                depth_flag, depth_val = 1.0, float(rng[i])
            else:
                depth_flag, depth_val = 0.0, -1.0
            feats.append([camsEN[i,0], camsEN[i,1], bsin[i], bcos[i], depth_flag, depth_val])
#             feats.append([camsEN[i,0], camsEN[i,1], bsin[i], bcos[i], depth_val])
            types.append(0)
        # hypotheses
        for k in range(len(hypsEN)):
            osm_flag_val = float(H.iloc[k][osm_col]) if have_osm else 0.0
            # feats.append([hypsEN[k,0], hypsEN[k,1], 0.0, 0.0, 0.0, 0.0])
#             feats.append([hypsEN[k,0], hypsEN[k,1], 0.0, 0.0, 0.0])
            feats.append([hypsEN[k,0], hypsEN[k,1], 0.0, 0.0, 0.0, osm_flag_val])
            types.append(1)

        x = torch.tensor(feats, dtype=torch.float)
        node_type = torch.tensor(types, dtype=torch.long)
        Nv, Nh = len(camsEN), len(hypsEN)

        # ----- edges (view -> hypothesis) with residuals -----
        rows, cols, attrs = [], [], []
        for i in range(Nv):
            for k in range(Nh):
                berr, dist = _bearing_err_and_dist(camsEN[i,0], camsEN[i,1], bsin[i], bcos[i],
                                                   hypsEN[k,0], hypsEN[k,1])
                range_err = -1.0
                if have_range and np.isfinite(rng[i]) and rng[i] > 0:
                    range_err = abs(dist - float(rng[i]))
                rows.append(i)
                cols.append(Nv + k)
                attrs.append([dist, berr, range_err])
#                 attrs.append([dist])
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr  = torch.tensor(attrs, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.node_type = node_type
        data.hypoth_idx = torch.arange(Nv, Nv + Nh, dtype=torch.long)
        data.hyps_EN = torch.tensor(hypsEN, dtype=torch.float)  # stash coords for inference
        data.object_id = oid

        # ----- label: index of hypothesis with MIN d_to_gt for this object -----
        if "d_to_gt" in H.columns:
            k_star = int(np.argmin(H["d_to_gt"].to_numpy(float)))
        else:
            # fallback (if GT not merged): nearest to GT from df (median) — optional
            if have_gt_xy:
                gtE = float(g["obj_E"].median()); gtN = float(g["obj_N"].median())
                d = np.linalg.norm(hypsEN - np.array([gtE, gtN])[None,:], axis=1)
                k_star = int(np.argmin(d))
            else:
                continue  # cannot label
        data.y = torch.tensor([k_star], dtype=torch.long)

        # (optional) keep GT for eval
        if have_gt_xy:
            data.gt_xy = torch.tensor([float(g["obj_E"].median()), float(g["obj_N"].median())], dtype=torch.float)

        data_list.append(data)

    return data_list





# WGS84 constants
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)
_WGS84_B = _WGS84_A * (1 - _WGS84_F)
_E2_PRIME = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_B**2

from sklearn.cluster import DBSCAN

def merge_close_predictions(pred_csv, out_csv="merged_predictions.csv", eps_meters=1.0, min_samples=1):
    """
    Merge multiple predicted points that are spatially close together using DBSCAN.
    eps_meters : maximum distance (in meters) for two predictions to be in the same cluster.
    min_samples : min number of samples per cluster (1 keeps all).
    """
    df = pred_csv.copy()
    if not {"pred_lat", "pred_lon"}.issubset(df.columns):
        raise ValueError("CSV must contain 'pred_lat' and 'pred_lon' columns")

    # --- approximate conversion: degrees -> meters (rough, fine for small areas) ---
    lat0 = np.deg2rad(df["pred_lat"].mean())
    m_per_deg_lat = 111_132.92
    m_per_deg_lon = 111_412.84 * np.cos(lat0)

    X = np.stack([
        df["pred_lon"] * m_per_deg_lon,
        df["pred_lat"] * m_per_deg_lat
    ], axis=1)

    # --- cluster close points ---
    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(X)
    df["cluster_id"] = clustering.labels_

    # --- merge each cluster by mean (or median) ---
    merged = (df.groupby("cluster_id", as_index=False)
                .agg({
                    "pred_lat": "mean",
                    "pred_lon": "mean",
                    "object_id": lambda x: list(x),
                }))

    merged.rename(columns={"pred_lat": "merged_lat", "pred_lon": "merged_lon"}, inplace=True)
    merged.to_csv(out_csv, index=False)
    print(f"[OK] Merged {len(df)} predictions into {len(merged)} clusters → {out_csv}")
    return merged


# ==========================================================
# MAIN FUNCTION inject noisy intersections
# ==========================================================

def filter_clean_hypotheses(hyps_df, dist_thresh=2.0):
    """
    Drop 'too perfect' clean hypotheses per object (d_to_gt < dist_thresh).
    Keeps the rest.
    """
    if "d_to_gt" not in hyps_df.columns:
        raise ValueError("hyps_df must contain 'd_to_gt' column for filtering.")

    # Group by object id and filter
    filtered = (
        hyps_df.groupby("trash_id", group_keys=False)
        .apply(lambda g: g[g["d_to_gt"] >= dist_thresh])
        .reset_index(drop=True)
    )
    dropped = len(hyps_df) - len(filtered)
    print(f"Filtered out {dropped} clean intersections with d_to_gt < {dist_thresh} m")
    return filtered


def inject_noisy_intersections(
    enu_csv: str,
    hyps_csv: str,
    output_csv: str = "hyps_with_noisy.csv",
    neighbor_radius: float = 25.0,
    cross_frac: float = 0.5,
    filter_clean_hyp = False
):
    """
    Adds noisy cross-object intersections to the hypotheses file, including lat/lon for QGIS.
    """
    print("Loading data...")
    enu_df = pd.read_csv(enu_csv)
    hyps_df = pd.read_csv(hyps_csv)

    required_cols = ["trash_id", "image_id", "image_lat", "image_lon", "trash_lat", "trash_lon", "obj_bearing"]
    for c in required_cols:
        if c not in enu_df.columns:
            raise ValueError(f"Missing column {c} in ENU CSV.")

    # Step 1: compute object centers
    obj_centers = enu_df.groupby("trash_id")[["trash_lat", "trash_lon"]].mean().reset_index()
    print((obj_centers))

    # Step 2: find neighbor object pairs
    pairs = []
    

    lat_ref = obj_centers["trash_lat"].mean()
    obj_centers["E"] = (obj_centers["trash_lon"] - obj_centers["trash_lon"].min()) * 111320 * cos(radians(lat_ref))
    obj_centers["N"] = (obj_centers["trash_lat"] - obj_centers["trash_lat"].min()) * 110540

    tree = cKDTree(obj_centers[["E", "N"]].to_numpy())
    pairs_idx = tree.query_pairs(r=neighbor_radius)
    pairs = [(obj_centers.trash_id.iloc[i], obj_centers.trash_id.iloc[j]) for i, j in pairs_idx]
    
    # for i, row1 in tqdm(obj_centers.iterrows()):
    #     for j, row2 in obj_centers.iterrows():
    #         if i >= j:
    #             continue
    #         dist = haversine(row1.trash_lat, row1.trash_lon, row2.trash_lat, row2.trash_lon)
    #         if 10 < dist < neighbor_radius:
    #             pairs.append((row1.trash_id, row2.trash_id))
    print(f"Found {len(pairs)} neighboring object pairs (within {neighbor_radius} m)")

    # Step 3: cross-view intersections
    noisy_records = []
    for a, b in tqdm(pairs):
        views_a = enu_df[enu_df.trash_id == a]
        views_b = enu_df[enu_df.trash_id == b]
        obj_a_lat, obj_a_lon = views_a.trash_lat.mean(), views_a.trash_lon.mean()
        obj_b_lat, obj_b_lon = views_b.trash_lat.mean(), views_b.trash_lon.mean()

        # Random subset of view pairs
        view_pairs = [(v1, v2) for _, v1 in views_a.iterrows() for _, v2 in views_b.iterrows()]
        np.random.shuffle(view_pairs)
        n_pairs = int(len(view_pairs) * cross_frac)
        view_pairs = view_pairs[:n_pairs]

        for v1, v2 in view_pairs:
            # ---- 1) Reference for local ENU ----
            ref_lat = (v1.image_lat + v2.image_lat) / 2
            ref_lon = (v1.image_lon + v2.image_lon) / 2

            # ---- 2) Convert image positions to ENU ----
            E1, N1 = latlon_to_ENU(v1.image_lat, v1.image_lon, ref_lat, ref_lon)
            E2, N2 = latlon_to_ENU(v2.image_lat, v2.image_lon, ref_lat, ref_lon)

            # ---- 3) Build bearing direction unit vectors ----
            theta1 = np.deg2rad(v1.obj_bearing)
            theta2 = np.deg2rad(v2.obj_bearing)
            d1 = np.array([math.sin(theta1), math.cos(theta1)])  # [dE, dN]
            d2 = np.array([math.sin(theta2), math.cos(theta2)])

            # ---- 4) Triangulate using your ENU-based function ----
            # tri_point, resid = closest_point_between_2d_rays_fast(
            #                         E1, N1, d1[0], d1[1], E2, N2, d2[0], d2[1],
            #                         MaxObjectDstFromCam=20)
            tri_point, resid = closest_point_between_2d_rays(
                P1=np.array([E1, N1]),
                d1=d1,
                P2=np.array([E2, N2]),
                d2=d2,
                MaxObjectDstFromCam=20
            )

            if np.isnan(tri_point).any() or resid == np.inf:
                continue

            tri_E, tri_N = tri_point

            # ---- 5) Convert back to lat/lon for QGIS ----
            tri_lat, tri_lon = ENU_to_latlon(tri_E, tri_N, ref_lat, ref_lon)

            # ---- 6) Compute distances to each true object (for label / filtering) ----
            d_to_a = haversine(tri_lat, tri_lon, obj_a_lat, obj_a_lon)
            d_to_b = haversine(tri_lat, tri_lon, obj_b_lat, obj_b_lon)

            # ---- 7) Keep if plausible ----
            if d_to_a <= neighbor_radius:
                noisy_records.append({
                    "trash_id": a,
                    "i": v1.image_id,
                    "j": v2.image_id,
                    "tri_E": tri_E,
                    "tri_N": tri_N,
                    "tri_lat": tri_lat,
                    "tri_lon": tri_lon,
                    "residual": resid,
                    "src": f"cross_{b}",
                    "jitter_k": np.nan,
                    "d_to_gt": d_to_a,
                    "is_noisy": 1
                })
            elif d_to_b <= neighbor_radius:
                noisy_records.append({
                    "trash_id": b,
                    "i": v1.image_id,
                    "j": v2.image_id,
                    "tri_E": tri_E,
                    "tri_N": tri_N,
                    "tri_lat": tri_lat,
                    "tri_lon": tri_lon,
                    "residual": resid,
                    "src": f"cross_{a}",
                    "jitter_k": np.nan,
                    "d_to_gt": d_to_b,
                    "is_noisy": 1
                })


    noisy_df = pd.DataFrame(noisy_records)
    print(f"Generated {len(noisy_df)} noisy intersections")

    # Step 4: merge with clean hypotheses
    if "is_noisy" not in hyps_df.columns:
        hyps_df["is_noisy"] = 0

    # Add lat/lon columns for true intersections if not present (optional for QGIS)
    if "tri_lat" not in hyps_df.columns or "tri_lon" not in hyps_df.columns:
        hyps_df["tri_lat"] = np.nan
        hyps_df["tri_lon"] = np.nan
    if filter_clean_hyp == True:
        filtered_hyps = filter_clean_hypotheses(hyps_df, dist_thresh=2.0)
        
        merged_hyps = pd.concat([filtered_hyps, noisy_df], ignore_index=True)
        merged_hyps.to_csv(output_csv, index=False)
    else:
        merged_hyps = pd.concat([hyps_df, noisy_df], ignore_index=True)
        merged_hyps.to_csv(output_csv, index=False)
        
    print(f"✅ Saved merged file with noisy + true hypotheses → {output_csv}")

    return merged_hyps


def plot_intersections_on_tiles(intersections_csv, tiles_dir="tiles",
                                image_pattern="bounds_tile_*.png", output_dir="tiles_with_points",
                                output_flag_csv="intersections_with_flags.csv"):
    """
    Plot intersection points on each OSM tile image.
    Points landing on black pixel are shown in red, others in green.
    
    Parameters
    ----------
    intersections_csv : str
        Path to CSV file with columns: lon, lat
    tiles_dir : str
        Directory where tile bound files (.txt) exist
    image_pattern : str
        Pattern for tile image names (e.g., 'bounds_tile_*.png')
    output_dir : str
        Directory where output plots with points will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    df_pts = pd.read_csv(intersections_csv)  # expects lon,lat columns
    df_pts["osm_flag"] = 0  # initialize
    print(f"Loaded {len(df_pts)} intersection points.")

    # get all tile text files (bounds)
    tile_txts = sorted(
        glob.glob(os.path.join(tiles_dir, "bounds_tile_*.txt")),
        key=lambda x: [int(re.search(r"tile_(\d+)_(\d+)", x).group(1)),
                       int(re.search(r"tile_(\d+)_(\d+)", x).group(2))]
    )

    # loop over each tile
    for tile_file in tile_txts:
        base = os.path.splitext(os.path.basename(tile_file))[0]
        img_file = os.path.join(tiles_dir, base + ".png")
        if not os.path.exists(img_file):
            print(f"⚠️ Missing image for {tile_file}")
            continue

        # read tile bounds
        lons, lats = [], []
        with open(tile_file) as f:
            for line in f:
                lon, lat = map(float, line.strip().split(","))
                lons.append(lon); lats.append(lat)
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        # west -= 0.0001
        # east += 0.0001
        # south -= 0.0001
        # north += 0.001


        # read image
        im = np.array(Image.open(img_file).convert("L"))
        H, W = im.shape

        # get points inside this tile
        mask = (
            (df_pts["tri_lon"] >= west) &
            (df_pts["tri_lon"] <= east) &
            (df_pts["tri_lat"] >= south) &
            (df_pts["tri_lat"] <= north)
        )
        df_tile = df_pts[mask]
        if df_tile.empty:
            continue

        # map lon/lat to pixel coordinates
        x_px = ((df_tile["tri_lon"] - west) / (east - west) * (W - 1)).astype(int)
        y_px = ((north - df_tile["tri_lat"]) / (north - south) * (H - 1)).astype(int)

        colors, flags = [], []
        for x, y in zip(x_px, y_px):
            pixel_val = im[y, x]
            if pixel_val <= 128:
                colors.append("red")   # on black area
                flags.append(1)
            else:
                colors.append("lime")  # on white area
                flags.append(0)
        # --- assign flags back to main df
        df_pts.loc[df_tile.index, "osm_flag"] = flags

        # plot
        fig, ax = plt.subplots(figsize=(8,8), dpi=150)
        ax.imshow(im, cmap="gray", origin="upper")
        ax.scatter(x_px, y_px, s=3, c=colors, alpha=0.8, edgecolors="none")
        ax.set_title(base)
        ax.axis("off")

        save_path = os.path.join(output_dir, base + "_with_points.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"✅ Saved: {save_path}")
        
    # --- save updated CSV
    df_pts["osm_flag"] = df_pts["osm_flag"].astype("Int64")
    df_pts.to_csv(output_flag_csv, index=False)
    print(f"\n✅ Updated CSV saved as {output_flag_csv}")
