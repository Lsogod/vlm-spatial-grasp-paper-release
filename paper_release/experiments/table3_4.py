# evaluate_static.py
# -*- coding: utf-8 -*-

import os
import json
import math
import glob
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2


# ----------------------------
# Utils: load + sanitize
# ----------------------------
def load_mask_u8(path: str) -> np.ndarray:
    """Load binary mask as uint8 {0,1}."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 0).astype(np.uint8)
    return m

def clamp_xy(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(W - 1, xi))
    yi = max(0, min(H - 1, yi))
    return xi, yi

def parse_coords(item: Any) -> Optional[Tuple[float, float]]:
    """Accept [x,y] or {'x':..,'y':..} or None."""
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return float(item[0]), float(item[1])
        except:
            return None
    if isinstance(item, dict) and "x" in item and "y" in item:
        try:
            return float(item["x"]), float(item["y"])
        except:
            return None
    return None


# ----------------------------
# Distance transform metric
# ----------------------------
@dataclass
class MaskMetric:
    hit: int                 # 1/0
    dist_px: float           # point -> nearest mask pixel (0 if inside)
    point_xy: Optional[Tuple[int, int]]

def compute_hit_and_distance(mask01: np.ndarray, coord_xy: Optional[Tuple[float, float]]) -> MaskMetric:
    """
    mask01: uint8 {0,1}, shape (H,W)
    coord_xy: (x,y) in the SAME coordinate system as mask (usually original image pixels)
    """
    H, W = mask01.shape[:2]
    if coord_xy is None:
        return MaskMetric(hit=0, dist_px=float("nan"), point_xy=None)

    x, y = coord_xy
    xi, yi = clamp_xy(x, y, W, H)

    inside = int(mask01[yi, xi] > 0)

    # distance to inside region:
    # distanceTransform expects non-zero as foreground.
    # We want distance from any pixel to nearest inside pixel => apply DT on (1 - mask)
    # Then distance at (xi,yi) tells how far it is from inside (0 if inside).
    outside = (1 - mask01).astype(np.uint8)
    dt = cv2.distanceTransform(outside, distanceType=cv2.DIST_L2, maskSize=3)  # float32
    dist = float(dt[yi, xi])

    return MaskMetric(hit=inside, dist_px=dist, point_xy=(xi, yi))


# ----------------------------
# Parse your prediction JSON
# ----------------------------
def load_prediction(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_grasp_place_coords(pred: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Try to extract grasp & place coords from your unified output:
    pred["results"][i]["ai_inference"]["final_action_plan"] = [{action, coords}, ...]
    coords could be [x,y] or null
    """
    # 兼容：如果 pred 本身就是单条结果，也能用
    results = pred.get("results", None)
    if isinstance(results, list) and len(results) > 0:
        # 这里你可以按 task_index 或只取第一条（每文件一图）来定
        item = results[0]
    else:
        item = pred

    plan = (((item.get("ai_inference") or {}).get("final_action_plan")) or
            ((item.get("ai_inference") or {}).get("action_plan")) or
            (item.get("final_action_plan")) or [])

    grasp_xy = None
    place_xy = None

    if isinstance(plan, list):
        for step in plan:
            act = str(step.get("action", "")).lower()
            c = parse_coords(step.get("coords", None))
            # 你可以按你自己的 action 命名规则改这里
            if grasp_xy is None and any(k in act for k in ["grasp", "pick", "抓", "夹"]):
                grasp_xy = c
            if place_xy is None and any(k in act for k in ["place", "put", "放", "摆"]):
                place_xy = c

    # 如果没有 action 关键词，但 plan 里就是两步：默认第1步 grasp 第2步 place
    if grasp_xy is None and place_xy is None and isinstance(plan, list) and len(plan) >= 1:
        grasp_xy = parse_coords(plan[0].get("coords", None)) if isinstance(plan[0], dict) else None
        if len(plan) >= 2 and isinstance(plan[1], dict):
            place_xy = parse_coords(plan[1].get("coords", None))

    return grasp_xy, place_xy


# ----------------------------
# Coordinate system adapter (optional)
# ----------------------------
def aligned_to_orig(coord_xy: Optional[Tuple[float, float]], orig_size: Tuple[int,int], aligned_size: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    """
    If your coords are on aligned image (W_new,H_new), map back to original (W_orig,H_orig).
    """
    if coord_xy is None:
        return None
    W_orig, H_orig = orig_size
    W_new, H_new = aligned_size
    sx = W_orig / float(W_new)
    sy = H_orig / float(H_new)
    return coord_xy[0] * sx, coord_xy[1] * sy


# ----------------------------
# Main evaluation
# ----------------------------
@dataclass
class SampleEval:
    name: str
    grasp_hit: int
    place_hit: int
    grasp_dist_px: float
    place_dist_px: float
    joint_hit: int

def evaluate_one(
    image_path: str,
    mask_grasp_path: str,
    mask_place_path: str,
    pred_json_path: str,
    coord_space: str = "orig",             # "orig" or "aligned"
    aligned_size: Optional[Tuple[int,int]] = None
) -> SampleEval:
    # sizes
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    mg = load_mask_u8(mask_grasp_path)
    mp = load_mask_u8(mask_place_path)
    if mg.shape != (H, W) or mp.shape != (H, W):
        raise ValueError(f"Mask size mismatch for {image_path}: img {(W,H)}, grasp {mg.shape[::-1]}, place {mp.shape[::-1]}")

    pred = load_prediction(pred_json_path)
    grasp_xy, place_xy = extract_grasp_place_coords(pred)

    if coord_space == "aligned":
        if aligned_size is None:
            raise ValueError("aligned_size is required when coord_space='aligned'")
        grasp_xy = aligned_to_orig(grasp_xy, orig_size=(W, H), aligned_size=aligned_size)
        place_xy = aligned_to_orig(place_xy, orig_size=(W, H), aligned_size=aligned_size)

    g = compute_hit_and_distance(mg, grasp_xy)
    p = compute_hit_and_distance(mp, place_xy)

    joint = int((g.hit == 1) and (p.hit == 1))

    name = os.path.splitext(os.path.basename(image_path))[0]
    return SampleEval(
        name=name,
        grasp_hit=g.hit,
        place_hit=p.hit,
        grasp_dist_px=g.dist_px,
        place_dist_px=p.dist_px,
        joint_hit=joint
    )


def aggregate(samples: List[SampleEval]) -> Dict[str, float]:
    # ignore nan distances when coords missing
    def _nanmean(arr):
        arr = np.array(arr, dtype=np.float32)
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    grasp_hit = [s.grasp_hit for s in samples]
    place_hit = [s.place_hit for s in samples]
    joint_hit = [s.joint_hit for s in samples]

    out = {
        "N": len(samples),
        "A_hit_grasp": float(np.mean(grasp_hit)) if len(samples) else float("nan"),
        "A_hit_place": float(np.mean(place_hit)) if len(samples) else float("nan"),
        "A_joint": float(np.mean(joint_hit)) if len(samples) else float("nan"),
        "D_spatial_grasp_px": _nanmean([s.grasp_dist_px for s in samples]),
        "D_spatial_place_px": _nanmean([s.place_dist_px for s in samples]),
    }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--masks_grasp_dir", required=True)
    ap.add_argument("--masks_place_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--ext", default="png")
    ap.add_argument("--coord_space", choices=["orig", "aligned"], default="orig")
    ap.add_argument("--aligned_w", type=int, default=None)
    ap.add_argument("--aligned_h", type=int, default=None)
    ap.add_argument("--save_csv", default=None)
    args = ap.parse_args()

    aligned_size = None
    if args.coord_space == "aligned":
        if args.aligned_w is None or args.aligned_h is None:
            raise ValueError("Need --aligned_w/--aligned_h when coord_space=aligned")
        aligned_size = (args.aligned_w, args.aligned_h)

    img_paths = sorted(glob.glob(os.path.join(args.images_dir, f"*.{args.ext}")))
    samples = []

    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        g_path = os.path.join(args.masks_grasp_dir, f"{name}.{args.ext}")
        p_path = os.path.join(args.masks_place_dir, f"{name}.{args.ext}")
        j_path = os.path.join(args.pred_dir, f"{name}.json")
        if not (os.path.exists(g_path) and os.path.exists(p_path) and os.path.exists(j_path)):
            # 允许缺失：跳过
            continue

        s = evaluate_one(
            image_path=img_path,
            mask_grasp_path=g_path,
            mask_place_path=p_path,
            pred_json_path=j_path,
            coord_space=args.coord_space,
            aligned_size=aligned_size
        )
        samples.append(s)

    agg = aggregate(samples)
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "grasp_hit", "place_hit", "joint_hit", "grasp_dist_px", "place_dist_px"])
            for s in samples:
                w.writerow([s.name, s.grasp_hit, s.place_hit, s.joint_hit, s.grasp_dist_px, s.place_dist_px])
        print(f"[Saved] {args.save_csv}")


if __name__ == "__main__":
    main()
