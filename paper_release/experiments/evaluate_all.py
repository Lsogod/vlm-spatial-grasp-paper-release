# evaluate_all.py
# -*- coding: utf-8 -*-
"""
Unified evaluator for:
- Chapter 3 (SVA anchors): Top-1 / Top-K / Dev (nearest distance), split by grasp/place
- Chapter 4 (SVP outputs): step_action_acc / step_coord_hit_rate / step_avg_spatial_dev /
                           step_joint_success_rate / traj_success

Usage examples:
1) Evaluate actions (your current evaluate_dataset.py logic):
   python evaluate_all.py --mode actions --npz_dir ../dist/100-129 --pred_json_dir ./ai_results_100-129

2) Evaluate anchors (Chapter 3):
   python evaluate_all.py --mode anchors --npz_dir ../dist/100-129 --anchor_dir ./anchors_100-129 --k_list 1,5,10,20

3) Evaluate both:
   python evaluate_all.py --mode both --npz_dir ../dist/100-129 --pred_json_dir ./ai_results_100-129 --anchor_dir ./anchors_100-129

Optional:
- --domain_map_json domain_map.json  (maps base_name -> domain like "GraspNet-Desk"/"Sim-Desk"/"Real-Desk")
- --dilate_r 2 (dilate GT regions to tolerate small annotation/pred errors)
- --require_instruction_match (drop samples whose instruction mismatch)
- --save_csv out.csv (dump summary table)
"""

import os
import json
import glob
import argparse
import csv
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
import cv2


# -----------------------------
# Utils: IO
# -----------------------------

def _safe_extract_json_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        elif value.ndim == 1 and value.size == 1:
            return value[0]
        else:
            return value.reshape(-1)[0]
    return value

def load_gt_from_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    if "trajectories" not in data.files:
        raise ValueError(f"npz missing trajectories: {npz_path}")
    traj_str = _safe_extract_json_string(data["trajectories"])
    traj_list = json.loads(traj_str)
    img = data["image"] if "image" in data.files else None
    H, W = (img.shape[:2] if img is not None else (None, None))
    return {"image": img, "H": H, "W": W, "trajectories": traj_list}

def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------
# GT masks (grasp/place)
# -----------------------------

def build_semantic_masks(H: int, W: int, traj: Dict[str, Any], dilate_r: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (grasp_mask_bool, place_mask_bool)"""
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
            # stored as [y, x]
            if coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
                ys = np.clip(coords_arr[:, 0], 0, H - 1)
                xs = np.clip(coords_arr[:, 1], 0, W - 1)
                target_mask[ys, xs] = 1

    # grasp: grasp_regions + fallback brush_regions
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

def compute_dist_map(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    """distanceTransform to nearest True pixel in mask.
       Return None if mask is empty.
    """
    if mask_bool.sum() == 0:
        return None
    inv = 1 - mask_bool.astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 5)


# -----------------------------
# Chapter 4: actions + coords evaluation
# -----------------------------

ACTION_TO_MASK_TYPE = {
    "move_to_grasp_pose": "grasp",
    "move_to_point":      "place",
}

def match_gt_traj(traj_list: List[Dict[str, Any]], task_item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool, Optional[int]]:
    task_idx = task_item.get("task_index", None)
    pred_instruction = task_item.get("original_instruction", "") or ""
    gt_traj = None
    matched_idx = None

    if isinstance(task_idx, int) and 0 <= task_idx < len(traj_list):
        gt_traj = traj_list[task_idx]
        matched_idx = task_idx
    else:
        # fallback: match by instruction
        for i, t in enumerate(traj_list):
            if (t.get("instruction", "") or "").strip() == pred_instruction.strip():
                gt_traj = t
                matched_idx = i
                break

    instruction_match = False
    if gt_traj is not None:
        gt_instruction = gt_traj.get("instruction", "") or ""
        instruction_match = (str(gt_instruction).strip() == str(pred_instruction).strip())

    return gt_traj, instruction_match, matched_idx

def evaluate_one_task_actions(H: int, W: int, traj_list: List[Dict[str, Any]], task_item: Dict[str, Any],
                              dilate_r: int = 0) -> Dict[str, Any]:
    gt_traj, instruction_match, matched_idx = match_gt_traj(traj_list, task_item)
    if gt_traj is None:
        return {"valid": False, "reason": "Trajectory not found"}

    ai_inference = task_item.get("ai_inference", {}) or {}
    status = ai_inference.get("status", "ok")
    if status == "failed":
        pred_components = []
    else:
        pred_components = ai_inference.get("action_components") or ai_inference.get("final_action_plan", []) or []

    gt_action_data = gt_traj.get("action", {})
    gt_actions = gt_action_data.get("actions", []) if isinstance(gt_action_data, dict) else list(gt_action_data)

    grasp_mask, place_mask = build_semantic_masks(H, W, gt_traj, dilate_r=dilate_r)
    dist_g = compute_dist_map(grasp_mask)
    dist_p = compute_dist_map(place_mask)

    max_steps = max(len(pred_components), len(gt_actions))
    if max_steps <= 0:
        return {"valid": False, "reason": "Empty steps"}

    correct_action_count = 0
    joint_success_count = 0
    coord_hit_count = 0
    num_steps_with_coord = 0
    spatial_dev_sum = 0.0

    # extra: split by grasp/place for coords
    coord_hit_g = coord_hit_p = 0
    coord_n_g = coord_n_p = 0
    dev_sum_g = dev_sum_p = 0.0

    step_results = []

    def in_bounds(px, py):
        return 0 <= px < W and 0 <= py < H

    for i in range(max_steps):
        if i < len(pred_components):
            comp = pred_components[i] or {}
            pred_action = comp.get("action", None)
            coords_raw = comp.get("coords", None)
        else:
            pred_action, coords_raw = None, None

        gt_action = gt_actions[i] if i < len(gt_actions) else None
        action_match = (str(pred_action) == str(gt_action))

        # decide which mask we should validate against
        target_type = None
        if gt_action in ACTION_TO_MASK_TYPE:
            target_type = ACTION_TO_MASK_TYPE[gt_action]
        elif pred_action in ACTION_TO_MASK_TYPE:
            target_type = ACTION_TO_MASK_TYPE[pred_action]

        needs_coord = (target_type is not None)

        hit = None
        spatial_dev = None
        coords_effective = None

        if needs_coord and pred_action is not None:
            num_steps_with_coord += 1
            hit = False

            if target_type == "grasp":
                curr_mask, curr_dist = grasp_mask, dist_g
                coord_n_g += 1
            else:
                curr_mask, curr_dist = place_mask, dist_p
                coord_n_p += 1

            if isinstance(coords_raw, (list, tuple)) and len(coords_raw) == 2:
                px, py = int(coords_raw[0]), int(coords_raw[1])
                coords_effective = (px, py)
                if in_bounds(px, py) and curr_mask is not None:
                    if curr_mask[py, px]:
                        hit = True
                        spatial_dev = 0.0
                    else:
                        hit = False
                        if curr_dist is None:
                            spatial_dev = float("inf")
                        else:
                            spatial_dev = float(curr_dist[py, px])
                            if np.isfinite(spatial_dev):
                                spatial_dev_sum += spatial_dev
                                if target_type == "grasp":
                                    dev_sum_g += spatial_dev
                                else:
                                    dev_sum_p += spatial_dev
                else:
                    hit = False
                    spatial_dev = float("inf")
            else:
                hit = False
                spatial_dev = float("inf")

            if hit:
                coord_hit_count += 1
                if target_type == "grasp":
                    coord_hit_g += 1
                else:
                    coord_hit_p += 1

        # joint success
        step_joint = False
        fail_reason = "OK"
        if not action_match:
            step_joint = False
            fail_reason = "Action Diff"
        else:
            if not needs_coord:
                step_joint = True
            else:
                if hit:
                    step_joint = True
                else:
                    step_joint = False
                    if target_type == "grasp" and (dist_g is None):
                        fail_reason = "No Grasp Region"
                    elif target_type == "place" and (dist_p is None):
                        fail_reason = "No Place Region"
                    else:
                        if spatial_dev is None or not np.isfinite(spatial_dev):
                            fail_reason = "Loc Dev (inf)"
                        else:
                            fail_reason = f"Loc Dev ({spatial_dev:.1f}px)"

        if action_match:
            correct_action_count += 1
        if step_joint:
            joint_success_count += 1

        step_results.append({
            "index": i,
            "pred_action": pred_action,
            "gt_action": gt_action,
            "target_type": target_type,
            "coords": coords_effective,
            "hit": hit,
            "spatial_dev": spatial_dev,
            "step_joint_success": step_joint,
            "fail_reason": fail_reason
        })

    # trajectory success (strict)
    traj_success = False
    if len(gt_actions) > 0 and len(pred_components) == len(gt_actions):
        if all(r["step_joint_success"] for r in step_results):
            traj_success = True

    denom_steps = max_steps if max_steps > 0 else 1
    metrics = {
        "num_steps": max_steps,
        "num_steps_with_coord": num_steps_with_coord,
        "step_action_acc": correct_action_count / denom_steps,
        "step_coord_hit_rate": (coord_hit_count / num_steps_with_coord) if num_steps_with_coord else 0.0,
        "step_avg_spatial_dev": (spatial_dev_sum / num_steps_with_coord) if num_steps_with_coord else 0.0,
        "step_joint_success_rate": joint_success_count / denom_steps,
        "traj_success": float(traj_success),

        # split grasp/place for coords
        "grasp_coord_hit_rate": (coord_hit_g / coord_n_g) if coord_n_g else 0.0,
        "place_coord_hit_rate": (coord_hit_p / coord_n_p) if coord_n_p else 0.0,
        "grasp_avg_dev": (dev_sum_g / coord_n_g) if coord_n_g else 0.0,
        "place_avg_dev": (dev_sum_p / coord_n_p) if coord_n_p else 0.0,
    }

    return {
        "valid": True,
        "matched_traj_index": matched_idx,
        "instruction_match": instruction_match,
        "metrics": metrics,
        "status": status,
    }


# -----------------------------
# Chapter 3: anchors evaluation
# -----------------------------

def normalize_points(points: Any) -> List[Tuple[int, int]]:
    """Accept [[x,y], ...] or [(x,y), ...]. Return list[(x,y)]"""
    if not points:
        return []
    out = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            out.append((int(p[0]), int(p[1])))
    return out

def pick_points_from_anchor_record(anchor_record: Dict[str, Any], strategy: str, stage: str) -> List[Tuple[int, int]]:
    """
    Supported formats:

    Format A (stage-specific):
      {
        "center": {"grasp": [[x,y],...], "place": [[x,y],...]},
        "random": {"grasp": [...], "place": [...]},
        "sva":    {"grasp": [...], "place": [...]}
      }

    Format B (shared for both stages):
      {
        "center": [[x,y],...],
        "random": [[x,y],...],
        "sva":    [[x,y],...]
      }
    """
    if strategy not in anchor_record:
        return []
    v = anchor_record[strategy]
    if isinstance(v, dict):
        return normalize_points(v.get(stage, []))
    return normalize_points(v)

def eval_points_topk(points: List[Tuple[int, int]], mask: np.ndarray, dist: Optional[np.ndarray], K: int) -> Optional[Dict[str, float]]:
    """
    Return metrics for given K:
      top1: first point hit
      topk: any of first K hit
      dev : 0 if hit else min distance among first K (inf if region empty or no valid points)
    Return None if GT region empty (dist is None and mask empty).
    """
    if mask is None or mask.sum() == 0 or dist is None:
        # treat as invalid sample for this stage
        return None

    H, W = mask.shape
    def in_bounds(x, y): return 0 <= x < W and 0 <= y < H

    pts = points[:K] if points else []
    if len(pts) == 0:
        return {"top1": 0.0, "topk": 0.0, "dev": float("inf")}

    hits = []
    devs = []
    for (x, y) in pts:
        if not in_bounds(x, y):
            hits.append(False)
            devs.append(float("inf"))
            continue
        if mask[y, x]:
            hits.append(True)
            devs.append(0.0)
        else:
            hits.append(False)
            devs.append(float(dist[y, x]))

    top1 = 1.0 if hits[0] else 0.0
    topk = 1.0 if any(hits) else 0.0
    dev = 0.0 if topk > 0 else float(np.min(devs)) if devs else float("inf")
    return {"top1": top1, "topk": topk, "dev": dev}

def evaluate_npz_anchors(npz_path: str,
                         anchor_path: str,
                         strategies: List[str],
                         k_list: List[int],
                         dilate_r: int = 0) -> List[Dict[str, Any]]:
    """
    One NPZ may contain multiple trajectories. Anchors are usually per-image (shared across trajectories),
    so we evaluate each trajectory against same anchors.
    """
    gt = load_gt_from_npz(npz_path)
    H, W = gt["H"], gt["W"]
    traj_list = gt["trajectories"]

    if H is None or W is None:
        return []

    anchor_record = load_json(anchor_path)
    if not anchor_record:
        return []

    reports = []
    for idx, traj in enumerate(traj_list):
        grasp_mask, place_mask = build_semantic_masks(H, W, traj, dilate_r=dilate_r)
        dist_g = compute_dist_map(grasp_mask)
        dist_p = compute_dist_map(place_mask)

        for strategy in strategies:
            pts_g = pick_points_from_anchor_record(anchor_record, strategy, "grasp")
            pts_p = pick_points_from_anchor_record(anchor_record, strategy, "place")
            # fallback: if stage-specific empty, try shared format
            if not pts_g:
                pts_g = normalize_points(anchor_record.get(strategy, []))
            if not pts_p:
                pts_p = normalize_points(anchor_record.get(strategy, []))

            for K in k_list:
                mg = eval_points_topk(pts_g, grasp_mask, dist_g, K)
                mp = eval_points_topk(pts_p, place_mask, dist_p, K)
                if mg is not None:
                    reports.append({
                        "traj_index": idx,
                        "stage": "grasp",
                        "strategy": strategy,
                        "K": K,
                        **mg
                    })
                if mp is not None:
                    reports.append({
                        "traj_index": idx,
                        "stage": "place",
                        "strategy": strategy,
                        "K": K,
                        **mp
                    })
    return reports


# -----------------------------
# Aggregation + printing
# -----------------------------

def mean_safe(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")

def load_domain_map(domain_map_json: Optional[str]) -> Dict[str, str]:
    """
    Mapping: base_name (npz/json filename without ext) -> domain label
    Example:
      {"100": "Sim-Desk", "101": "Sim-Desk", "2001": "Real-Desk"}
    """
    if not domain_map_json:
        return {}
    d = load_json(domain_map_json)
    return d if isinstance(d, dict) else {}

def get_domain(base_name: str, domain_map: Dict[str, str]) -> str:
    return domain_map.get(base_name, "ALL")

def save_summary_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize_actions(action_reports: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = [
        "traj_success",
        "step_action_acc",
        "step_coord_hit_rate",
        "step_avg_spatial_dev",
        "step_joint_success_rate",
        "grasp_coord_hit_rate",
        "place_coord_hit_rate",
        "grasp_avg_dev",
        "place_avg_dev",
    ]
    out = {}
    for k in keys:
        out[k] = mean_safe([r["metrics"].get(k, None) for r in action_reports])
    return out

def summarize_anchors(anchor_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    out = {}
    for k in ["top1", "topk", "dev"]:
        out[k] = mean_safe([r.get(k, None) for r in anchor_rows])
    return out


# -----------------------------
# Main
# -----------------------------

def parse_k_list(s: str) -> List[int]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    out = [k for k in out if k > 0]
    return sorted(list(set(out))) if out else [10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["actions", "anchors", "both"], default="actions")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--pred_json_dir", default=None, help="folder of ai_results_xxx/*.json")
    ap.add_argument("--anchor_dir", default=None, help="folder of anchors_xxx/*.json (one per base_name)")
    ap.add_argument("--domain_map_json", default=None, help="optional: base_name -> domain")
    ap.add_argument("--dilate_r", type=int, default=0)
    ap.add_argument("--k_list", default="10", help="for anchors mode, e.g. 1,5,10,20")
    ap.add_argument("--strategies", default="center,random,sva", help="anchors strategies, comma-separated")
    ap.add_argument("--require_instruction_match", action="store_true")
    ap.add_argument("--save_csv", default=None, help="dump summary rows to csv")
    args = ap.parse_args()

    domain_map = load_domain_map(args.domain_map_json)
    k_list = parse_k_list(args.k_list)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not npz_files:
        print("❌ No NPZ files found.")
        return

    summary_rows: List[Dict[str, Any]] = []

    # ---------------- actions ----------------
    action_reports_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    if args.mode in ["actions", "both"]:
        if not args.pred_json_dir:
            print("❌ --pred_json_dir is required for actions mode.")
            return
        json_files = sorted(glob.glob(os.path.join(args.pred_json_dir, "*.json")))
        if not json_files:
            print("❌ No prediction JSON files found.")
            return

        print(f"[ACTIONS] Processing {len(json_files)} prediction files...")
        for json_path in json_files:
            base = os.path.splitext(os.path.basename(json_path))[0]
            npz_path = os.path.join(args.npz_dir, base + ".npz")
            if not os.path.exists(npz_path):
                continue

            gt = load_gt_from_npz(npz_path)
            H, W = gt["H"], gt["W"]
            traj_list = gt["trajectories"]
            if H is None or W is None:
                continue

            root_json = load_json(json_path)
            if not root_json:
                continue
            results_list = root_json.get("results", []) or []

            dom = get_domain(base, domain_map)
            action_reports_by_domain.setdefault(dom, [])

            for task_item in results_list:
                rep = evaluate_one_task_actions(H, W, traj_list, task_item, dilate_r=args.dilate_r)
                if not rep.get("valid", False):
                    continue
                if args.require_instruction_match and (not rep.get("instruction_match", False)):
                    continue
                action_reports_by_domain[dom].append(rep)

        # print summary
        all_reports = sum(action_reports_by_domain.values(), [])
        if not all_reports:
            print("❌ No valid action reports.")
        else:
            print("\n" + "=" * 60)
            print("📊 ACTIONS SUMMARY (Chapter 4/5 style)")
            print("=" * 60)

            for dom, reps in sorted(action_reports_by_domain.items(), key=lambda x: x[0]):
                if not reps:
                    continue
                s = summarize_actions(reps)
                print(f"\n[Domain: {dom}]  N={len(reps)}")
                print(f"  Traj Success         : {s['traj_success']:.3f}")
                print(f"  Step Action Acc       : {s['step_action_acc']:.3f}")
                print(f"  Step Coord Hit Rate   : {s['step_coord_hit_rate']:.3f}")
                print(f"  Step Avg Spatial Dev  : {s['step_avg_spatial_dev']:.2f} px")
                print(f"  Step Joint Success    : {s['step_joint_success_rate']:.3f}")
                print(f"  Grasp Hit / Dev       : {s['grasp_coord_hit_rate']:.3f} / {s['grasp_avg_dev']:.2f}px")
                print(f"  Place Hit / Dev       : {s['place_coord_hit_rate']:.3f} / {s['place_avg_dev']:.2f}px")

                summary_rows.append({
                    "mode": "actions",
                    "domain": dom,
                    "N": len(reps),
                    **s
                })

    # ---------------- anchors ----------------
    anchor_rows_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    if args.mode in ["anchors", "both"]:
        if not args.anchor_dir:
            print("❌ --anchor_dir is required for anchors mode.")
            return
        print(f"\n[ANCHORS] Processing {len(npz_files)} NPZ files with anchors...")
        for npz_path in npz_files:
            base = os.path.splitext(os.path.basename(npz_path))[0]
            anchor_path = os.path.join(args.anchor_dir, base + ".json")
            if not os.path.exists(anchor_path):
                continue
            dom = get_domain(base, domain_map)
            anchor_rows_by_domain.setdefault(dom, [])
            rows = evaluate_npz_anchors(npz_path, anchor_path, strategies, k_list, dilate_r=args.dilate_r)
            anchor_rows_by_domain[dom].extend(rows)

        all_rows = sum(anchor_rows_by_domain.values(), [])
        if not all_rows:
            print("❌ No valid anchor rows.")
        else:
            print("\n" + "=" * 60)
            print("📊 ANCHORS SUMMARY (Chapter 3 style)")
            print("=" * 60)

            # group: domain -> stage -> strategy -> K
            for dom, rows in sorted(anchor_rows_by_domain.items(), key=lambda x: x[0]):
                if not rows:
                    continue
                print(f"\n[Domain: {dom}]  rows={len(rows)}")
                for stage in ["grasp", "place"]:
                    for strategy in strategies:
                        for K in k_list:
                            sub = [r for r in rows if r["stage"] == stage and r["strategy"] == strategy and r["K"] == K]
                            if not sub:
                                continue
                            s = summarize_anchors(sub)
                            print(f"  {stage:5s} | {strategy:8s} | K={K:2d}  Top1={s['top1']:.3f}  TopK={s['topk']:.3f}  Dev={s['dev']:.2f}px")
                            summary_rows.append({
                                "mode": "anchors",
                                "domain": dom,
                                "stage": stage,
                                "strategy": strategy,
                                "K": K,
                                "N": len(sub),
                                **s
                            })

    if args.save_csv:
        save_summary_csv(args.save_csv, summary_rows)
        print(f"\n✅ Saved summary CSV -> {args.save_csv}")

if __name__ == "__main__":
    main()
