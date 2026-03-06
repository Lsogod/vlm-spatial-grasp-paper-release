# evaluate_table4_2_strict.py
# -*- coding: utf-8 -*-
"""
Generate Table 4.2 (Strict Supplementary Evaluation) for multiple prediction folders.

Metrics (micro-average, consistent with evaluate_dataset.py logic):
- InstrMatch : (#tasks instruction_match=True) / (#tasks)
- TrajSucc   : (#tasks traj_success=True) / (#tasks)
- ActAcc     : (sum correct_action_count) / (sum metrics_denom=max_steps)
- CoordHit   : (sum coord_hit_count) / (sum num_steps_with_coord)
- JointSucc  : (sum joint_success_count) / (sum metrics_denom=max_steps)
- AvgDev(px) : (sum spatial_dev_sum) / (sum num_steps_with_coord)
  NOTE: spatial_dev_sum only accumulates finite deviations, exactly following your code.

Assumptions:
- NPZ name matches JSON name: <base>.npz <-> <base>.json
- JSON schema:
  { "results": [ { "task_index": int, "original_instruction": str,
                  "ai_inference": { "status": "...", "action_components"/"final_action_plan": [...] } }, ... ] }
- NPZ contains "trajectories" (json string) with each traj having:
  - instruction
  - action/actions
  - grasp_regions / brush_regions / place_regions (painted_coords)
"""

import os
import json
import glob
import csv
import argparse
import numpy as np
import cv2

# =======================
# Configure your methods
# =======================
DEFAULT_NPZ_DIR = r"../dist/all"

SETTINGS = [
    ("E1-GPT-5 (Free-coordinate)",   r"./gpt-5_results"),
    ("E1-GPT-4o (Free-coordinate)",  r"./gpt-4o_results"),
    ("E2-Qwen3-VL-Plus (Trained Coord)",  r"./qwen_plus_results_newGT"),
    ("E2-Qwen3-VL-Flash (Trained Coord)", r"./qwen_flash_results_newGT"),
    ("E3-GPT-5 (SVP-Full, Ours)",     r"./ai_results"),
    ("E3-GPT-4o (SVP-Full, Ours)",    r"./ai_results_4o"),
]
# =======================


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
        "image_shape": data["image"].shape[:2],   # (H, W)
        "trajectories": traj_list,
    }


def build_semantic_masks(H, W, traj, dilate_r=0):
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
    H, W = mask_bool.shape
    if mask_bool.sum() > 0:
        inv_mask = 1 - mask_bool.astype(np.uint8)
        return cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    else:
        return np.full((H, W), float("inf"), dtype=np.float32)


def evaluate_one_task_strict(npz_gt, task_item, dilate_r=0):
    """
    A strict evaluator that mirrors your evaluate_dataset.py core logic,
    but returns raw counters for micro-averaging + mismatch debug info.
    """
    H, W = npz_gt["image_shape"]
    traj_list = npz_gt["trajectories"]

    task_idx = task_item.get("task_index")
    pred_instruction = task_item.get("original_instruction", "")

    gt_traj = None
    if task_idx is not None and 0 <= task_idx < len(traj_list):
        gt_traj = traj_list[task_idx]
    else:
        # fallback by instruction string (same as your code)
        for t in traj_list:
            if str(t.get("instruction", "")).strip() == str(pred_instruction).strip():
                gt_traj = t
                break

    if gt_traj is None:
        return None  # invalid

    gt_instruction = gt_traj.get("instruction", "")
    instruction_match = str(gt_instruction).strip() == str(pred_instruction).strip()

    ai_inference = task_item.get("ai_inference", {})
    if ai_inference.get("status") == "failed":
        pred_components = []
    else:
        pred_components = (
            ai_inference.get("action_components")
            or ai_inference.get("final_action_plan", [])
            or []
        )

    gt_action_data = gt_traj.get("action", {})
    gt_actions = (
        gt_action_data.get("actions", [])
        if isinstance(gt_action_data, dict)
        else list(gt_action_data)
    )

    grasp_mask, place_mask = build_semantic_masks(H, W, gt_traj, dilate_r=dilate_r)
    dist_map_grasp = compute_dist_map(grasp_mask)
    dist_map_place = compute_dist_map(place_mask)

    max_steps = max(len(pred_components), len(gt_actions))

    correct_action_count = 0
    coord_hit_count = 0
    joint_success_count = 0
    num_steps_with_coord = 0
    spatial_dev_sum = 0.0

    ACTION_TO_MASK_TYPE = {
        "move_to_grasp_pose": "grasp",
        "move_to_point": "place",
    }

    step_joint_success_flags = []

    for i in range(max_steps):
        if i < len(pred_components):
            comp = pred_components[i] if isinstance(pred_components[i], dict) else {}
            pred_action = comp.get("action", None)
            coords_raw = comp.get("coords", None)
        else:
            pred_action, coords_raw = None, None

        gt_action = gt_actions[i] if i < len(gt_actions) else None
        action_match = str(pred_action) == str(gt_action)

        target_mask_type = None
        if gt_action in ACTION_TO_MASK_TYPE:
            target_mask_type = ACTION_TO_MASK_TYPE[gt_action]
        elif pred_action in ACTION_TO_MASK_TYPE:
            target_mask_type = ACTION_TO_MASK_TYPE[pred_action]

        needs_coord = target_mask_type is not None

        hit = None
        spatial_dev = None

        if needs_coord and pred_action is not None:
            num_steps_with_coord += 1
            hit = False

            if target_mask_type == "grasp":
                curr_mask = grasp_mask
                curr_dist = dist_map_grasp
            else:
                curr_mask = place_mask
                curr_dist = dist_map_place

            if isinstance(coords_raw, (list, tuple)) and len(coords_raw) == 2:
                px, py = int(coords_raw[0]), int(coords_raw[1])
                if 0 <= px < W and 0 <= py < H:
                    if curr_mask[py, px]:
                        hit = True
                        spatial_dev = 0.0
                    else:
                        hit = False
                        spatial_dev = float(curr_dist[py, px])
                        if np.isfinite(spatial_dev):
                            spatial_dev_sum += spatial_dev
                else:
                    hit = False
            else:
                hit = False

        # joint success logic (same as your code)
        if not action_match:
            step_joint = False
        else:
            if not needs_coord:
                step_joint = True
            else:
                step_joint = True if hit else False

        if action_match:
            correct_action_count += 1
        if hit:
            coord_hit_count += 1
        if step_joint:
            joint_success_count += 1

        step_joint_success_flags.append(step_joint)

    metrics_denom = max_steps if max_steps > 0 else 1

    traj_success = False
    if metrics_denom > 0 and len(gt_actions) > 0:
        if len(pred_components) == len(gt_actions):
            if all(step_joint_success_flags):
                traj_success = True

    return {
        "instruction_match": bool(instruction_match),
        "gt_instruction": str(gt_instruction),
        "pred_instruction": str(pred_instruction),
        "traj_success": bool(traj_success),

        "correct_action_count": int(correct_action_count),
        "joint_success_count": int(joint_success_count),
        "coord_hit_count": int(coord_hit_count),

        "num_steps_with_coord": int(num_steps_with_coord),
        "metrics_denom": int(metrics_denom),
        "spatial_dev_sum": float(spatial_dev_sum),
    }


def evaluate_one_setting(npz_dir, pred_json_dir, dilate_r=0, dump_mismatch=False):
    json_files = sorted(glob.glob(os.path.join(pred_json_dir, "*.json")))
    if not json_files:
        print(f"⚠️ No JSON files found in: {pred_json_dir}")
        return None

    # micro-average accumulators
    tasks = 0
    instr_match = 0
    traj_succ = 0

    sum_correct_action = 0
    sum_joint_succ = 0
    sum_coord_hit = 0

    sum_steps = 0
    sum_coord_steps = 0
    sum_spatial_dev = 0.0

    mismatch_cases = []

    npz_matched = 0

    for json_path in json_files:
        base = os.path.basename(json_path).replace(".json", "")
        npz_path = os.path.join(npz_dir, f"{base}.npz")
        if not os.path.exists(npz_path):
            continue

        try:
            npz_gt = load_gt_from_npz(npz_path)
        except Exception:
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                root = json.load(f)
        except Exception:
            continue

        res_list = root.get("results", [])
        if not isinstance(res_list, list):
            continue

        npz_matched += 1

        for task_item in res_list:
            out = evaluate_one_task_strict(npz_gt, task_item, dilate_r=dilate_r)
            if out is None:
                continue

            tasks += 1
            if out["instruction_match"]:
                instr_match += 1
            else:
                if dump_mismatch:
                    mismatch_cases.append({
                        "json_file": os.path.basename(json_path),
                        "npz_file": os.path.basename(npz_path),
                        "task_index": task_item.get("task_index"),
                        "gt_instruction": out["gt_instruction"],
                        "pred_instruction": out["pred_instruction"],
                    })

            if out["traj_success"]:
                traj_succ += 1

            sum_correct_action += out["correct_action_count"]
            sum_joint_succ += out["joint_success_count"]
            sum_coord_hit += out["coord_hit_count"]

            sum_steps += out["metrics_denom"]
            sum_coord_steps += out["num_steps_with_coord"]
            sum_spatial_dev += out["spatial_dev_sum"]

    if tasks == 0:
        print(f"⚠️ No valid tasks in: {pred_json_dir}")
        return None

    # micro-averaged metrics
    InstrMatch = instr_match / tasks if tasks else float("nan")
    TrajSucc = traj_succ / tasks if tasks else float("nan")
    ActAcc = sum_correct_action / sum_steps if sum_steps else float("nan")
    JointSucc = sum_joint_succ / sum_steps if sum_steps else float("nan")
    CoordHit = sum_coord_hit / sum_coord_steps if sum_coord_steps else float("nan")
    AvgDev = sum_spatial_dev / sum_coord_steps if sum_coord_steps else float("nan")

    return {
        "tasks": tasks,
        "npz_matched": npz_matched,
        "InstrMatch": InstrMatch,
        "TrajSucc": TrajSucc,
        "ActAcc": ActAcc,
        "CoordHit": CoordHit,
        "JointSucc": JointSucc,
        "AvgDev": AvgDev,
        "mismatch_cases": mismatch_cases,
    }


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{x*100:.2f}%"

def px(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--out_csv", type=str, default="table4_2_strict.csv")
    parser.add_argument("--dump_mismatch", action="store_true", help="dump mismatch cases to a json file per method")
    parser.add_argument("--dilate_r", type=int, default=0, help="optional dilation radius, keep 0 to match your current eval")
    args = parser.parse_args()

    rows = []

    print("\nTable 4.2 (Strict Supplementary Evaluation) - Full\n" + "=" * 118)
    print(f"{'Method':<35} | {'InstrMatch ↑':>11} {'TrajSucc ↑':>11} {'ActAcc ↑':>10} {'CoordHit ↑':>11} {'JointSucc ↑':>12} {'AvgDev ↓':>10}")
    print("-" * 118)

    for label, pred_dir in SETTINGS:
        res = evaluate_one_setting(
            npz_dir=args.npz_dir,
            pred_json_dir=pred_dir,
            dilate_r=args.dilate_r,
            dump_mismatch=args.dump_mismatch
        )

        if res is None:
            print(f"{label:<35} | {'-':>11} {'-':>11} {'-':>10} {'-':>11} {'-':>12} {'-':>10}")
            rows.append([label, "", "", "", "", "", ""])
            continue

        print(
            f"{label:<35} | {pct(res['InstrMatch']):>11} {pct(res['TrajSucc']):>11} {pct(res['ActAcc']):>10} "
            f"{pct(res['CoordHit']):>11} {pct(res['JointSucc']):>12} {px(res['AvgDev']):>10}"
        )

        rows.append([
            label,
            res["InstrMatch"], res["TrajSucc"], res["ActAcc"], res["CoordHit"], res["JointSucc"], res["AvgDev"]
        ])

        if args.dump_mismatch:
            out_m = f"mismatch_{label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace(',', '')}.json"
            with open(out_m, "w", encoding="utf-8") as f:
                json.dump(res["mismatch_cases"], f, ensure_ascii=False, indent=2)
            print(f"   ↳ mismatch dumped: {out_m}  (n={len(res['mismatch_cases'])})")

    print("=" * 118)

    # Save CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "InstrMatch", "TrajSucc", "ActAcc", "CoordHit", "JointSucc", "AvgDev_px"])
        for r in rows:
            w.writerow(r)

    print(f"\n✅ Saved: {args.out_csv}")
    print("   - Rates in [0,1], AvgDev in pixels (px).")
    print("   - Aggregation = micro-average over all tasks/steps (recommended for Table 4.2).")


if __name__ == "__main__":
    main()
