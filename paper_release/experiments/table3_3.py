# table3-3_viz.py
# -*- coding: utf-8 -*-

import os
import json
import glob
import random
import argparse
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

DEFAULT_NPZ_DIR = r"../dist/all"
DEFAULT_PRED_JSON_DIR = r"./ai_results_4o"

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# Domain split by filename leading index
# 0-99 / 100-129 / 130-159
# =========================
def parse_leading_index(filename_no_ext: str):
    m = re.match(r"^(\d+)", filename_no_ext)
    if not m:
        return None
    try:
        return int(m.group(1))
    except:
        return None

def domain_from_index(idx):
    if idx is None:
        return "unknown"
    if 0 <= idx <= 99:
        return "graspnet"
    if 100 <= idx <= 129:
        return "real"
    if 130 <= idx <= 159:
        return "sim"
    return "unknown"

def domain_label(dom: str):
    return {
        "graspnet": "GraspNet-Desk",
        "real": "Real-Desk",
        "sim": "Sim-Desk",
        "all": "ALL",
        "unknown": "UNKNOWN",
    }.get(dom, dom)

# =========================
# NPZ helpers
# =========================
def _safe_extract_json_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        elif value.ndim == 1 and value.size == 1:
            return value[0]
        else:
            return value.reshape(-1)[0]
    return value

def load_gt_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "trajectories" not in data.files:
        raise ValueError(f"npz missing trajectories: {npz_path}")
    traj_str = _safe_extract_json_string(data["trajectories"])
    traj_list = json.loads(traj_str)
    return {
        "npz": data,
        "image": data["image"],
        "image_shape": data["image"].shape[:2],
        "trajectories": traj_list,
    }

def build_semantic_masks(H, W, traj, dilate_r=0):
    grasp_mask = np.zeros((H, W), dtype=np.uint8)
    place_mask = np.zeros((H, W), dtype=np.uint8)

    kernel = None
    if dilate_r > 0:
        k = dilate_r * 2 + 1
        kernel = np.ones((k, k), np.uint8)

    def draw_regions(region_list, target_mask):
        if not region_list or not isinstance(region_list, list):
            return
        for br in region_list:
            coords = br.get("painted_coords", [])
            if not coords:
                continue
            coords_arr = np.array(coords, dtype=np.int32)
            if coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
                valid_y = np.clip(coords_arr[:, 0], 0, H - 1)
                valid_x = np.clip(coords_arr[:, 1], 0, W - 1)
                target_mask[valid_y, valid_x] = 1

    # grasp: new + legacy
    draw_regions(traj.get("grasp_regions", []), grasp_mask)
    draw_regions(traj.get("brush_regions", []), grasp_mask)
    # place
    draw_regions(traj.get("place_regions", []), place_mask)

    if kernel is not None:
        if grasp_mask.sum() > 0:
            grasp_mask = cv2.dilate(grasp_mask, kernel)
        if place_mask.sum() > 0:
            place_mask = cv2.dilate(place_mask, kernel)

    return grasp_mask.astype(bool), place_mask.astype(bool)

def compute_dist_map(mask_bool):
    if mask_bool.sum() > 0:
        inv = 1 - mask_bool.astype(np.uint8)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    H, W = mask_bool.shape
    return np.full((H, W), float("inf"), dtype=np.float32)

# =========================
# instance bank: instance_masks + results(id->mask_index)
# =========================
def load_instance_bank(npz_handle):
    if "instance_masks" not in npz_handle.files:
        return None, [], {}

    instance_masks = np.array(npz_handle["instance_masks"])
    res_raw = npz_handle.get("results", None)

    results = []
    if res_raw is not None:
        try:
            results = json.loads(str(res_raw))
        except:
            try:
                s = _safe_extract_json_string(res_raw)
                results = json.loads(str(s))
            except:
                results = []

    id2midx = {}
    for r in results:
        try:
            rid = int(r.get("id"))
            midx = int(r.get("mask_index"))
            id2midx[rid] = midx
        except:
            continue

    return instance_masks, results, id2midx

def instance_mask_by_id(instance_masks, id2midx, target_id):
    if target_id is None:
        return None
    try:
        tid = int(target_id)
    except:
        return None
    if tid not in id2midx:
        return None
    midx = int(id2midx[tid])
    if not (0 <= midx < instance_masks.shape[0]):
        return None

    m = instance_masks[midx]
    if m.dtype == np.bool_:
        return m
    mx = float(np.max(m)) if m.size else 0.0
    if mx <= 1.0:
        return (m > 0)
    return (m > 127)

# =========================
# Baseline point generators
# =========================
def baseline_point_center(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return (int(np.round(xs.mean())), int(np.round(ys.mean())))

def baseline_point_dtmax(mask_bool):
    if mask_bool.sum() == 0:
        return None
    mask_u8 = mask_bool.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_in = dist.copy()
    dist_in[~mask_bool] = -1
    idx = int(np.argmax(dist_in))
    if dist_in.flat[idx] <= 0:
        return None
    y, x = np.unravel_index(idx, dist_in.shape)
    return (int(x), int(y))

def baseline_point_random_in_mask(mask_bool, rng):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    j = int(rng.integers(0, len(xs)))
    return (int(xs[j]), int(ys[j]))

def baseline_point(mask_bool, method, rng):
    if method == "center":
        return baseline_point_center(mask_bool)
    if method == "dtmax":
        return baseline_point_dtmax(mask_bool)
    if method == "random":
        return baseline_point_random_in_mask(mask_bool, rng)
    raise ValueError(method)

# =========================
# GT matching + action extraction
# =========================
ACTION_TO_TYPE = {
    "move_to_grasp_pose": "grasp",
    "move_to_point": "place",
}

def match_gt_traj(traj_list, task_item):
    task_idx = task_item.get("task_index", None)
    pred_instruction = task_item.get("original_instruction", "")

    if task_idx is not None:
        try:
            ti = int(task_idx)
            if 0 <= ti < len(traj_list):
                return traj_list[ti], ti, pred_instruction
        except:
            pass

    for i, t in enumerate(traj_list):
        if str(t.get("instruction", "")).strip() == str(pred_instruction).strip():
            return t, i, pred_instruction

    return None, None, pred_instruction

def get_pred_plan(task_item):
    ai = task_item.get("ai_inference", {}) or {}
    if ai.get("status") == "failed":
        return []
    plan = ai.get("final_action_plan", None)
    if isinstance(plan, list) and plan:
        return plan
    plan2 = ai.get("action_components", None)
    return plan2 if isinstance(plan2, list) else []

def get_gt_actions(gt_traj):
    gt_action_data = gt_traj.get("action", {})
    if isinstance(gt_action_data, dict):
        return gt_action_data.get("actions", [])
    if isinstance(gt_action_data, list):
        return gt_action_data
    return []

# =========================
# Scoring helpers (hit/dev)
# =========================
def step_hit_dev(px, py, mask_bool, dist_map):
    H, W = mask_bool.shape
    if px is None or py is None:
        return False, float("inf")
    if not (0 <= px < W and 0 <= py < H):
        return False, float("inf")
    if mask_bool[py, px]:
        return True, 0.0
    d = float(dist_map[py, px])
    return False, d

# =========================
# Evaluate one task for a method
# method in: center/dtmax/random/sva
# =========================
def evaluate_one_task(npz_path, task_item, method="dtmax", dilate_r=0, repeat=1):
    gt = load_gt_from_npz(npz_path)
    H, W = gt["image_shape"]
    traj_list = gt["trajectories"]
    npz_handle = gt["npz"]

    gt_traj, matched_idx, pred_instruction = match_gt_traj(traj_list, task_item)
    if gt_traj is None:
        return None

    grasp_gt, place_gt = build_semantic_masks(H, W, gt_traj, dilate_r=dilate_r)
    dist_g = compute_dist_map(grasp_gt)
    dist_p = compute_dist_map(place_gt)

    pred_plan = get_pred_plan(task_item)
    gt_actions = get_gt_actions(gt_traj)

    instance_masks, results, id2midx = load_instance_bank(npz_handle)

    max_steps = max(len(pred_plan), len(gt_actions))
    if max_steps <= 0:
        return None

    g_total = 0
    p_total = 0
    g_hit = 0
    p_hit = 0
    g_dev_sum = 0.0
    p_dev_sum = 0.0

    joint_total = max_steps
    joint_ok = 0

    diag = {
        "no_instance_bank": 0,
        "missing_mask_for_id": 0,
        "missing_coords_in_sva": 0,
        "dtmax_none": 0,
    }

    if instance_masks is None or len(id2midx) == 0:
        diag["no_instance_bank"] = 1

    # deterministic seed per file (optional)
    base_seed = RANDOM_SEED
    try:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        idx0 = parse_leading_index(base)
        if idx0 is not None:
            base_seed += idx0 * 1000
    except:
        pass

    for i in range(max_steps):
        pred = pred_plan[i] if i < len(pred_plan) else {}
        gt_act = gt_actions[i] if i < len(gt_actions) else None

        pred_action = pred.get("action", None)
        action_match = (str(pred_action) == str(gt_act))

        target_type = None
        if gt_act in ACTION_TO_TYPE:
            target_type = ACTION_TO_TYPE[gt_act]
        elif pred_action in ACTION_TO_TYPE:
            target_type = ACTION_TO_TYPE[pred_action]

        needs_coord = (target_type in ("grasp", "place"))

        px = py = None
        if needs_coord:
            if method == "sva":
                coords = pred.get("coords", None)
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    px, py = int(coords[0]), int(coords[1])
                else:
                    diag["missing_coords_in_sva"] += 1
            else:
                tid = pred.get("target_id", None)
                m = instance_mask_by_id(instance_masks, id2midx, tid) if (instance_masks is not None) else None
                if m is None:
                    diag["missing_mask_for_id"] += 1
                else:
                    best_hit = False
                    best_dev = float("inf")
                    best_pt = None
                    for rj in range(max(1, repeat)):
                        rng = np.random.default_rng(base_seed + i * 997 + rj * 131)
                        pt = baseline_point(m, method, rng)
                        if pt is None and method == "dtmax":
                            diag["dtmax_none"] += 1
                            continue
                        if pt is None:
                            continue
                        x, y = pt
                        if target_type == "grasp":
                            hit, dev = step_hit_dev(x, y, grasp_gt, dist_g)
                        else:
                            hit, dev = step_hit_dev(x, y, place_gt, dist_p)
                        if hit and (not best_hit):
                            best_hit = True
                            best_dev = 0.0
                            best_pt = pt
                        elif (hit == best_hit) and (np.isfinite(dev) and dev < best_dev):
                            best_dev = dev
                            best_pt = pt
                    if best_pt is not None:
                        px, py = int(best_pt[0]), int(best_pt[1])

        hit = None
        dev = None
        if needs_coord:
            if target_type == "grasp":
                g_total += 1
                hit, dev = step_hit_dev(px, py, grasp_gt, dist_g)
                if hit:
                    g_hit += 1
                if np.isfinite(dev):
                    g_dev_sum += float(dev)
            else:
                p_total += 1
                hit, dev = step_hit_dev(px, py, place_gt, dist_p)
                if hit:
                    p_hit += 1
                if np.isfinite(dev):
                    p_dev_sum += float(dev)

        step_joint = False
        if action_match:
            if not needs_coord:
                step_joint = True
            else:
                step_joint = bool(hit)
        if step_joint:
            joint_ok += 1

    A_hit_g = (g_hit / g_total) if g_total else 0.0
    A_hit_p = (p_hit / p_total) if p_total else 0.0
    D_g = (g_dev_sum / g_total) if g_total else 0.0
    D_p = (p_dev_sum / p_total) if p_total else 0.0
    A_joint = (joint_ok / joint_total) if joint_total else 0.0

    return {
        "A_hit_g": A_hit_g,
        "D_spatial_g": D_g,
        "A_hit_p": A_hit_p,
        "D_spatial_p": D_p,
        "A_joint": A_joint,
        "diag": diag,
    }

# =========================
# Run for a given (domain, method)
# =========================
def run_one(npz_dir, pred_json_dir, domain, method, dilate_r=0, repeat=1):
    all_json = sorted(glob.glob(os.path.join(pred_json_dir, "*.json")))

    selected = []
    for p in all_json:
        base = os.path.splitext(os.path.basename(p))[0]
        idx = parse_leading_index(base)
        dom = domain_from_index(idx)
        if dom == domain:
            selected.append((idx, p))
    selected.sort(key=lambda t: (t[0] if t[0] is not None else 10**9, t[1]))

    reports = []
    diag_no_bank = 0
    diag_missing_mask = 0
    diag_dtmax_none = 0
    diag_missing_coords_sva = 0

    for idx, jp in selected:
        base_name = os.path.basename(jp).replace(".json", "")
        npz_path = os.path.join(npz_dir, base_name + ".npz")
        if not os.path.exists(npz_path):
            continue
        try:
            root = json.load(open(jp, "r", encoding="utf-8"))
        except:
            continue

        for task_item in (root.get("results", []) or []):
            rep = evaluate_one_task(npz_path, task_item, method=method, dilate_r=dilate_r, repeat=repeat)
            if rep is None:
                continue
            reports.append(rep)
            d = rep["diag"]
            if d.get("no_instance_bank", 0):
                diag_no_bank += 1
            if d.get("missing_mask_for_id", 0) > 0:
                diag_missing_mask += 1
            if d.get("dtmax_none", 0) > 0:
                diag_dtmax_none += 1
            if d.get("missing_coords_in_sva", 0) > 0:
                diag_missing_coords_sva += 1

    if not reports:
        return {
            "num_reports": 0,
            "A_hit_g": 0.0, "D_spatial_g": 0.0,
            "A_hit_p": 0.0, "D_spatial_p": 0.0,
            "A_joint": 0.0,
            "diag_no_bank": diag_no_bank,
            "diag_missing_mask": diag_missing_mask,
            "diag_dtmax_none": diag_dtmax_none,
            "diag_missing_coords_sva": diag_missing_coords_sva,
        }

    def avg(key):
        vals = [r[key] for r in reports if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "num_reports": len(reports),
        "A_hit_g": avg("A_hit_g"),
        "D_spatial_g": avg("D_spatial_g"),
        "A_hit_p": avg("A_hit_p"),
        "D_spatial_p": avg("D_spatial_p"),
        "A_joint": avg("A_joint"),
        "diag_no_bank": diag_no_bank,
        "diag_missing_mask": diag_missing_mask,
        "diag_dtmax_none": diag_dtmax_none,
        "diag_missing_coords_sva": diag_missing_coords_sva,
    }

# =========================
# Plot helpers
# =========================
def save_bar_chart(groups, series, values_dict, title, ylabel, save_path, as_percent=False):
    """
    groups: list[str] e.g. ["GraspNet-Desk","Real-Desk","Sim-Desk"]
    series: list[str] e.g. ["center","dtmax","random","sva"]
    values_dict[(group, series)] -> float
    """
    x = np.arange(len(groups))
    width = 0.18 if len(series) >= 4 else 0.25

    plt.figure(figsize=(10, 5))
    for j, s in enumerate(series):
        y = [values_dict.get((g, s), 0.0) for g in groups]
        offset = (j - (len(series)-1)/2) * width
        plt.bar(x + offset, y, width=width, label=s)

    plt.xticks(x, groups)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def write_csv(rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

# =========================
# Batch summarize + visualize
# =========================
def run_batch(npz_dir, pred_json_dir, out_dir, dilate_r=0, random_repeat=20):
    domains = ["graspnet", "real", "sim"]
    methods = ["center", "dtmax", "random", "sva"]

    rows = []
    print(f"Batch Summary -> {out_dir}")
    for dom in domains:
        for m in methods:
            rep = run_one(
                npz_dir=npz_dir,
                pred_json_dir=pred_json_dir,
                domain=dom,
                method=m,
                dilate_r=dilate_r,
                repeat=(random_repeat if m == "random" else 1)
            )
            row = {
                "domain": domain_label(dom),
                "method": m,
                "num_reports": rep["num_reports"],
                "A_hit_g": rep["A_hit_g"],
                "D_spatial_g": rep["D_spatial_g"],
                "A_hit_p": rep["A_hit_p"],
                "D_spatial_p": rep["D_spatial_p"],
                "A_joint": rep["A_joint"],
                "diag_no_bank": rep["diag_no_bank"],
                "diag_missing_mask": rep["diag_missing_mask"],
                "diag_dtmax_none": rep["diag_dtmax_none"],
                "diag_missing_coords_sva": rep["diag_missing_coords_sva"],
            }
            rows.append(row)
            print(f"{row['domain']:>12} | {m:>6} | "
                  f"A_hit_g={row['A_hit_g']:.2%} Dg={row['D_spatial_g']:.1f} | "
                  f"A_hit_p={row['A_hit_p']:.2%} Dp={row['D_spatial_p']:.1f} | "
                  f"A_joint={row['A_joint']:.2%} | n={row['num_reports']}")

    # save csv
    csv_path = os.path.join(out_dir, "summary.csv")
    write_csv(rows, csv_path)
    print(f"[SAVE] CSV -> {csv_path}")

    # prepare charts
    group_names = [domain_label(d) for d in domains]
    series = methods

    def build_values(key):
        vd = {}
        for r in rows:
            g = r["domain"]
            s = r["method"]
            vd[(g, s)] = float(r[key])
        return vd

    # hit/joint charts (0..1)
    save_bar_chart(group_names, series, build_values("A_hit_g"),
                   title="A_hit_g (Grasp Hit Rate)", ylabel="rate",
                   save_path=os.path.join(out_dir, "bar_A_hit_g.png"))
    save_bar_chart(group_names, series, build_values("A_hit_p"),
                   title="A_hit_p (Place Hit Rate)", ylabel="rate",
                   save_path=os.path.join(out_dir, "bar_A_hit_p.png"))
    save_bar_chart(group_names, series, build_values("A_joint"),
                   title="A_joint (Joint Success Rate)", ylabel="rate",
                   save_path=os.path.join(out_dir, "bar_A_joint.png"))

    # deviation charts (px)
    save_bar_chart(group_names, series, build_values("D_spatial_g"),
                   title="D_spatial_g (Grasp Spatial Deviation)", ylabel="px",
                   save_path=os.path.join(out_dir, "bar_D_spatial_g.png"))
    save_bar_chart(group_names, series, build_values("D_spatial_p"),
                   title="D_spatial_p (Place Spatial Deviation)", ylabel="px",
                   save_path=os.path.join(out_dir, "bar_D_spatial_p.png"))

    print(f"[SAVE] Charts -> {out_dir}")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, default=DEFAULT_NPZ_DIR)
    ap.add_argument("--pred_json_dir", type=str, default=DEFAULT_PRED_JSON_DIR)

    ap.add_argument("--mode", type=str, default="batch", choices=["batch"],
                    help="batch: run 3 domains x 4 methods and plot bar charts")

    ap.add_argument("--out_dir", type=str, default="out_summary")
    ap.add_argument("--dilate_r", type=int, default=0)
    ap.add_argument("--random_repeat", type=int, default=20, help="repeat for random baseline in batch")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_batch(
        npz_dir=args.npz_dir,
        pred_json_dir=args.pred_json_dir,
        out_dir=args.out_dir,
        dilate_r=args.dilate_r,
        random_repeat=args.random_repeat
    )
