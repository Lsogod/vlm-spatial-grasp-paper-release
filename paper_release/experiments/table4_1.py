# table4_full_from_evaluate_dataset.py
# -*- coding: utf-8 -*-
"""
Generate Table 4-1 (Full) - Oracle-Action
Based on the EXACT evaluation logic in your evaluate_dataset.py.

What it computes (Full = merge all domains, micro-average over steps):
- A_hit,g      : grasp coord hit rate over grasp-required steps
- D_spatial,g  : mean spatial deviation (px) over grasp-required steps with finite dist only
- A_hit,p      : place coord hit rate over place-required steps
- D_spatial,p  : mean spatial deviation (px) over place-required steps with finite dist only
- A_joint      : micro-average over all coord-required steps = (#hit_g + #hit_p) / (Ng + Np)

Important:
- Pred steps are taken from:
    pred_components = ai_inference.get("action_components") or ai_inference.get("final_action_plan", [])
  (same as your evaluate_dataset.py)
- Coords must be list/tuple of length 2: [x, y]
- Action->mask type mapping is the same:
    move_to_grasp_pose -> grasp
    move_to_point      -> place
- If coords missing/invalid/outside => hit=False
- If GT region empty => dist_map is INF everywhere; we count those steps in denominator (hit stays False),
  and we track how many such steps exist. For D_spatial we only average finite distances (same spirit as you did).
"""

import os
import json
import glob
import csv
import argparse
import numpy as np
import cv2


# -------------------------
# Default settings (edit if needed)
# -------------------------
DEFAULT_NPZ_DIR = r"../dist/all"

SETTINGS = [
    ("E1-GPT-5 (Free-coordinate)",   r"./gpt-5_results"),
    ("E1-GPT-4o (Free-coordinate)",  r"./gpt-4o_results"),
    ("E2-Qwen3-VL-Plus (Trained)",   r"./qwen_plus_results_newGT"),
    ("E2-Qwen3-VL-Flash (Trained)",  r"./qwen_flash_results_newGT"),
    ("E3-GPT-5 (SVP-Full, Ours)",    r"./ai_results"),
    ("E3-GPT-4o (SVP-Full, Ours)",   r"./ai_results_4o"),
]
# -------------------------


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

    raw_traj = data["trajectories"]
    traj_str = _safe_extract_json_string(raw_traj)
    traj_list = json.loads(traj_str)

    return {
        "image": data["image"],
        "image_shape": data["image"].shape[:2],
        "trajectories": traj_list,
    }


def build_semantic_masks(H, W, traj, dilate_r=0):
    # copied from your evaluate_dataset.py (same behavior)
    grasp_mask = np.zeros((H, W), dtype=np.uint8)
    place_mask = np.zeros((H, W), dtype=np.uint8)

    kernel = None
    if dilate_r > 0:
        ksize = dilate_r * 2 + 1
        kernel = np.ones((ksize, ksize), np.uint8)

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

    draw_regions(traj.get("grasp_regions", []), grasp_mask)
    draw_regions(traj.get("brush_regions", []), grasp_mask)
    draw_regions(traj.get("place_regions", []), place_mask)

    if kernel is not None:
        if grasp_mask.sum() > 0:
            grasp_mask = cv2.dilate(grasp_mask, kernel)
        if place_mask.sum() > 0:
            place_mask = cv2.dilate(place_mask, kernel)

    return grasp_mask.astype(bool), place_mask.astype(bool)


def compute_dist_map(mask_bool):
    # copied from your evaluate_dataset.py (same behavior)
    H, W = mask_bool.shape
    if mask_bool.sum() > 0:
        inv_mask = 1 - mask_bool.astype(np.uint8)
        return cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    else:
        return np.full((H, W), float("inf"), dtype=np.float32)


ACTION_TO_MASK_TYPE = {
    "move_to_grasp_pose": "grasp",
    "move_to_point": "place",
}


def get_pred_components(task_item):
    # EXACT as your evaluate_dataset.py
    ai_inference = task_item.get("ai_inference", {}) or {}
    if ai_inference.get("status") == "failed":
        return []
    return ai_inference.get("action_components") or ai_inference.get("final_action_plan", []) or []


def match_gt_traj(traj_list, task_item):
    # EXACT matching strategy as your evaluate_dataset.py
    task_idx = task_item.get("task_index")
    pred_instruction = task_item.get("original_instruction", "")

    gt_traj = None
    if task_idx is not None and 0 <= task_idx < len(traj_list):
        gt_traj = traj_list[task_idx]
        return gt_traj

    # fallback: instruction string match
    for t in traj_list:
        if str(t.get("instruction", "")).strip() == str(pred_instruction).strip():
            gt_traj = t
            break
    return gt_traj


def eval_one_task_for_table(gt, task_item, dilate_r=0):
    """
    Returns per-step contributions for grasp/place coord-required steps:
    - counts, hits, dev sums (finite only), and diagnostic counts.
    """
    H, W = gt["image_shape"]
    traj_list = gt["trajectories"]

    gt_traj = match_gt_traj(traj_list, task_item)
    if gt_traj is None:
        return None  # skip

    grasp_mask, place_mask = build_semantic_masks(H, W, gt_traj, dilate_r=dilate_r)
    dist_g = compute_dist_map(grasp_mask)
    dist_p = compute_dist_map(place_mask)

    pred_components = get_pred_components(task_item)

    # In Oracle-Action for your Table 4-1: we treat coord-required steps by ACTION_TO_MASK_TYPE.
    # We do NOT need to compare action correctness here; just use pred actions to decide stage
    # because your Table defines hit/dev for grasp and place.
    # (If you want to force GT action stage instead, you can swap to gt_traj.get("action").)
    # We'll follow your evaluate_dataset.py preference: target_mask_type can be derived from pred_action.
    out = {
        "Ng": 0, "Np": 0,
        "hit_g": 0, "hit_p": 0,
        "dev_sum_g": 0.0, "dev_sum_p": 0.0,
        "dev_cnt_g": 0, "dev_cnt_p": 0,
        "missing_coord_g": 0, "missing_coord_p": 0,
        "oob_coord_g": 0, "oob_coord_p": 0,
        "no_region_g": 0, "no_region_p": 0,  # GT region empty => dist map inf everywhere
    }

    for comp in pred_components:
        if not isinstance(comp, dict):
            continue
        pred_action = comp.get("action", None)
        coords_raw = comp.get("coords", None)

        if pred_action not in ACTION_TO_MASK_TYPE:
            continue  # only evaluate coord-required actions

        ttype = ACTION_TO_MASK_TYPE[pred_action]
        needs = True  # by definition

        if ttype == "grasp":
            out["Ng"] += 1
            curr_mask, curr_dist = grasp_mask, dist_g
        else:
            out["Np"] += 1
            curr_mask, curr_dist = place_mask, dist_p

        # GT region empty?
        if curr_mask.sum() == 0:
            if ttype == "grasp":
                out["no_region_g"] += 1
            else:
                out["no_region_p"] += 1

        # coords validity
        if not (isinstance(coords_raw, (list, tuple)) and len(coords_raw) == 2):
            if ttype == "grasp":
                out["missing_coord_g"] += 1
            else:
                out["missing_coord_p"] += 1
            continue

        px, py = int(coords_raw[0]), int(coords_raw[1])
        if not (0 <= px < W and 0 <= py < H):
            if ttype == "grasp":
                out["oob_coord_g"] += 1
            else:
                out["oob_coord_p"] += 1
            continue

        # hit / dev
        if curr_mask[py, px]:
            if ttype == "grasp":
                out["hit_g"] += 1
            else:
                out["hit_p"] += 1
            # dev=0 counts as finite
            if ttype == "grasp":
                out["dev_sum_g"] += 0.0
                out["dev_cnt_g"] += 1
            else:
                out["dev_sum_p"] += 0.0
                out["dev_cnt_p"] += 1
        else:
            d = float(curr_dist[py, px])
            if np.isfinite(d):
                if ttype == "grasp":
                    out["dev_sum_g"] += d
                    out["dev_cnt_g"] += 1
                else:
                    out["dev_sum_p"] += d
                    out["dev_cnt_p"] += 1
            # miss already implied (no hit increment)

    return out


def safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def fmt_pct(x):
    return "-" if (x is None or not np.isfinite(x)) else f"{x*100:.2f}%"


def fmt_px(x):
    return "-" if (x is None or not np.isfinite(x)) else f"{x:.2f}"


def evaluate_one_setting(npz_dir, pred_dir, dilate_r=0):
    json_files = sorted(glob.glob(os.path.join(pred_dir, "*.json")))
    if not json_files:
        return None, {"err": f"no json files in {pred_dir}"}

    agg = {
        "Ng": 0, "Np": 0,
        "hit_g": 0, "hit_p": 0,
        "dev_sum_g": 0.0, "dev_sum_p": 0.0,
        "dev_cnt_g": 0, "dev_cnt_p": 0,
        "missing_coord_g": 0, "missing_coord_p": 0,
        "oob_coord_g": 0, "oob_coord_p": 0,
        "no_region_g": 0, "no_region_p": 0,
        "npz_matched": 0,
        "tasks_seen": 0,
        "tasks_used": 0,
    }

    for json_path in json_files:
        base = os.path.basename(json_path).replace(".json", "")
        npz_path = os.path.join(npz_dir, f"{base}.npz")
        if not os.path.exists(npz_path):
            continue

        try:
            gt = load_gt_from_npz(npz_path)
        except Exception:
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                root = json.load(f)
        except Exception:
            continue

        agg["npz_matched"] += 1
        results = root.get("results", []) or []
        for task_item in results:
            agg["tasks_seen"] += 1
            out = eval_one_task_for_table(gt, task_item, dilate_r=dilate_r)
            if out is None:
                continue
            agg["tasks_used"] += 1
            for k in out:
                agg[k] += out[k]

    # Compute metrics
    A_hit_g = safe_div(agg["hit_g"], agg["Ng"])
    A_hit_p = safe_div(agg["hit_p"], agg["Np"])
    D_g = safe_div(agg["dev_sum_g"], agg["dev_cnt_g"])
    D_p = safe_div(agg["dev_sum_p"], agg["dev_cnt_p"])
    A_joint = safe_div(agg["hit_g"] + agg["hit_p"], agg["Ng"] + agg["Np"])

    metrics = {
        "A_hit_g": A_hit_g,
        "D_spatial_g": D_g,
        "A_hit_p": A_hit_p,
        "D_spatial_p": D_p,
        "A_joint": A_joint,
    }
    return metrics, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, default=DEFAULT_NPZ_DIR, help="NPZ directory")
    ap.add_argument("--out_csv", type=str, default="table4_full.csv", help="output CSV")
    ap.add_argument("--dilate_r", type=int, default=0, help="mask dilation radius (same as evaluate_dataset.py)")
    args = ap.parse_args()

    print("\nTable 4-1 (Full) - Oracle-Action")
    print("=" * 110)
    print(f"{'Method':<35} | {'A_hit,g ↑':>10} {'D_g ↓':>10} | {'A_hit,p ↑':>10} {'D_p ↓':>10} | {'A_joint ↑':>10}")
    print("-" * 110)

    rows = []
    diag_rows = []

    for label, pred_dir in SETTINGS:
        m, diag = evaluate_one_setting(args.npz_dir, pred_dir, dilate_r=args.dilate_r)
        if m is None:
            print(f"{label:<35} | {'-':>10} {'-':>10} | {'-':>10} {'-':>10} | {'-':>10}")
            rows.append([label, "", "", "", "", ""])
            continue

        print(f"{label:<35} | {fmt_pct(m['A_hit_g']):>10} {fmt_px(m['D_spatial_g']):>10} | "
              f"{fmt_pct(m['A_hit_p']):>10} {fmt_px(m['D_spatial_p']):>10} | {fmt_pct(m['A_joint']):>10}")

        rows.append([label, m["A_hit_g"], m["D_spatial_g"], m["A_hit_p"], m["D_spatial_p"], m["A_joint"]])

        # Diagnostics (super useful for explaining zeros)
        diag_rows.append([
            label,
            diag["npz_matched"], diag["tasks_seen"], diag["tasks_used"],
            diag["Ng"], diag["missing_coord_g"], diag["oob_coord_g"], diag["no_region_g"],
            diag["Np"], diag["missing_coord_p"], diag["oob_coord_p"], diag["no_region_p"],
        ])

    print("=" * 110)

    # Save metrics CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "A_hit_g", "D_spatial_g_px", "A_hit_p", "D_spatial_p_px", "A_joint"])
        for r in rows:
            w.writerow(r)

    # Save diagnostics CSV
    diag_csv = os.path.splitext(args.out_csv)[0] + "_diag.csv"
    with open(diag_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "Method",
            "npz_matched", "tasks_seen", "tasks_used",
            "Ng", "missing_coord_g", "oob_coord_g", "no_region_g",
            "Np", "missing_coord_p", "oob_coord_p", "no_region_p",
        ])
        for r in diag_rows:
            w.writerow(r)

    print(f"\n✅ Saved: {args.out_csv}")
    print(f"✅ Saved diagnostics: {diag_csv}")
    print("   - A_hit_* and A_joint are in [0,1], D_spatial_* in pixels.")


if __name__ == "__main__":
    main()
