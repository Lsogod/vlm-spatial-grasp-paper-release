import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_leading_index(filename_no_ext: str) -> Optional[int]:
    m = re.match(r"^(\d+)", filename_no_ext)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def collect_npz_by_index(npz_dir: str) -> Dict[int, str]:
    by_idx = {}
    for name in sorted(os.listdir(npz_dir)):
        if not name.lower().endswith(".npz"):
            continue
        base = os.path.splitext(name)[0]
        idx = parse_leading_index(base)
        if idx is None:
            continue
        if idx not in by_idx:
            by_idx[idx] = os.path.join(npz_dir, name)
    return by_idx


def pick_four_indices(start: int, end: int) -> List[int]:
    vals = np.linspace(start, end, 4)
    out = []
    for v in vals:
        iv = int(round(float(v)))
        iv = max(start, min(end, iv))
        if iv not in out:
            out.append(iv)
    while len(out) < 4:
        candidate = end - (4 - len(out)) + 1
        if candidate not in out and start <= candidate <= end:
            out.append(candidate)
        else:
            break
    return sorted(out)[:4]


def parse_indices_arg(text: str, start: int, end: int) -> Optional[List[int]]:
    if not text:
        return None
    items = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            v = int(p)
        except Exception:
            continue
        if start <= v <= end:
            items.append(v)
    # keep order + dedupe
    uniq = []
    for v in items:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) == 4:
        return uniq
    return None


def to_json_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1 and value.size == 1:
            return value[0]
        return value.reshape(-1)[0]
    return value


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return str(value).strip()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return normalize_text(value.item())
        if value.size == 1:
            return normalize_text(value.reshape(-1)[0])
        try:
            return " / ".join([normalize_text(x) for x in value.reshape(-1).tolist() if normalize_text(x)])
        except Exception:
            return str(value).strip()
    if isinstance(value, (list, tuple)):
        parts = [normalize_text(x) for x in value]
        parts = [x for x in parts if x]
        return " / ".join(parts)
    return str(value).strip()


def extract_dataset_instruction(npz_handle) -> str:
    if "instruction" not in npz_handle.files:
        return ""
    raw = npz_handle["instruction"]
    txt = normalize_text(raw)
    # 如果是 JSON 字符串，尝试提取可读文本
    if txt.startswith("[") or txt.startswith("{"):
        try:
            obj = json.loads(txt)
            if isinstance(obj, str):
                return obj.strip()
            if isinstance(obj, list):
                joined = " / ".join([normalize_text(x) for x in obj if normalize_text(x)])
                if joined:
                    return joined
            if isinstance(obj, dict):
                for k in ("instruction", "text", "prompt"):
                    if k in obj and normalize_text(obj[k]):
                        return normalize_text(obj[k])
        except Exception:
            pass
    return txt


def _extract_action_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        if isinstance(value.get("actions"), list):
            return [normalize_text(x) for x in value.get("actions", []) if normalize_text(x)]
        return []
    if isinstance(value, list):
        return [normalize_text(x) for x in value if normalize_text(x)]
    text = normalize_text(value)
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
            return _extract_action_list(obj)
        except Exception:
            pass
    return []


def extract_dataset_action(npz_handle) -> str:
    if "action" not in npz_handle.files:
        return ""
    raw = npz_handle["action"]
    seq = _extract_action_list(raw)
    return " -> ".join(seq)


def extract_traj_action(traj: dict) -> str:
    seq = _extract_action_list(traj.get("action"))
    return " -> ".join(seq)


def draw_regions_to_mask(
    target_mask: np.ndarray,
    region_list: Sequence[dict],
):
    if not isinstance(region_list, list):
        return
    h, w = target_mask.shape
    for reg in region_list:
        coords = reg.get("painted_coords", [])
        if not coords:
            continue
        arr = np.asarray(coords, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        ys = np.clip(arr[:, 0], 0, h - 1)
        xs = np.clip(arr[:, 1], 0, w - 1)
        target_mask[ys, xs] = 255


def build_grasp_place_masks(image_hw: Tuple[int, int], traj: dict) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_hw
    grasp = np.zeros((h, w), dtype=np.uint8)
    place = np.zeros((h, w), dtype=np.uint8)

    draw_regions_to_mask(grasp, traj.get("grasp_regions", []))
    draw_regions_to_mask(grasp, traj.get("brush_regions", []))
    draw_regions_to_mask(place, traj.get("place_regions", []))
    return grasp, place


def alpha_blend_mask(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = image_bgr.copy()
    if mask_u8.sum() == 0:
        return out
    color_layer = np.zeros_like(out, dtype=np.uint8)
    color_layer[:, :] = np.array(color_bgr, dtype=np.uint8)
    mask_bool = mask_u8 > 0
    out[mask_bool] = cv2.addWeighted(out[mask_bool], 1.0 - alpha, color_layer[mask_bool], alpha, 0)
    return out


def draw_contours(image_bgr: np.ndarray, mask_u8: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int = 2):
    if mask_u8.sum() == 0:
        return image_bgr
    out = image_bgr.copy()
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color_bgr, thickness=thickness)
    return out


def pick_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return [""]
    lines = []
    curr = ""
    for ch in text:
        trial = curr + ch
        w = draw.textbbox((0, 0), trial, font=font)[2]
        if w <= max_width or not curr:
            curr = trial
        else:
            lines.append(curr)
            curr = ch
    if curr:
        lines.append(curr)
    return lines


def draw_header_and_legend(
    image_bgr: np.ndarray,
    instruction: str,
    traj_instruction: str,
    gt_action: str,
    traj_action: str,
    split_name: str,
) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_title = pick_font(30)
    font_body = pick_font(24)

    w, h = pil.size
    instruction = instruction or "(empty instruction)"
    text_lines = []
    text_lines += wrap_text_to_width(draw, f"Dataset-Instruction: {instruction}", font_body, max_width=w - 30)
    if traj_instruction and traj_instruction.strip() != instruction.strip():
        text_lines += wrap_text_to_width(draw, f"Trajectory-Instruction: {traj_instruction}", font_body, max_width=w - 30)
    if gt_action:
        text_lines += wrap_text_to_width(draw, f"GT-Action: {gt_action}", font_body, max_width=w - 30)
    if traj_action and traj_action.strip() != gt_action.strip():
        text_lines += wrap_text_to_width(draw, f"Trajectory-Action: {traj_action}", font_body, max_width=w - 30)

    line_h = 34
    box_h = 16 + line_h * len(text_lines) + 56
    box_h = min(box_h, int(0.45 * h))

    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ovd = ImageDraw.Draw(overlay)
    ovd.rectangle([(0, 0), (w, box_h)], fill=(0, 0, 0, 140))

    y = 10
    for i, line in enumerate(text_lines):
        font = font_title if i == 0 else font_body
        ovd.text((14, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h

    legend_y = box_h - 44
    ovd.rectangle([(14, legend_y), (38, legend_y + 24)], fill=(0, 255, 0, 255))
    ovd.text((46, legend_y), "Grasp/Brush", fill=(255, 255, 255, 255), font=font_body)
    ovd.rectangle([(250, legend_y), (274, legend_y + 24)], fill=(255, 0, 0, 255))
    ovd.text((282, legend_y), "Place", fill=(255, 255, 255, 255), font=font_body)

    merged = Image.alpha_composite(pil.convert("RGBA"), overlay)
    return cv2.cvtColor(np.array(merged.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_one_visual(npz_path: str, traj_index: int, split_name: str) -> Tuple[np.ndarray, str, str]:
    data = np.load(npz_path, allow_pickle=True)
    image = np.array(data["image"])
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unexpected image shape: {image.shape} ({npz_path})")
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    traj_raw = to_json_string(data["trajectories"])
    trajs = json.loads(traj_raw)
    if not isinstance(trajs, list) or len(trajs) == 0:
        raise ValueError(f"empty trajectories in {npz_path}")
    ti = max(0, min(traj_index, len(trajs) - 1))
    traj = trajs[ti]

    grasp_mask, place_mask = build_grasp_place_masks(image_bgr.shape[:2], traj)
    vis = alpha_blend_mask(image_bgr, grasp_mask, color_bgr=(0, 255, 0), alpha=0.35)
    vis = alpha_blend_mask(vis, place_mask, color_bgr=(255, 0, 0), alpha=0.35)
    vis = draw_contours(vis, grasp_mask, color_bgr=(0, 200, 0), thickness=2)
    vis = draw_contours(vis, place_mask, color_bgr=(0, 0, 255), thickness=2)

    sample_name = os.path.splitext(os.path.basename(npz_path))[0]
    dataset_instruction = extract_dataset_instruction(data)
    dataset_action = extract_dataset_action(data)
    traj_instruction = str(traj.get("instruction", ""))
    traj_action = extract_traj_action(traj)
    instruction = dataset_instruction or traj_instruction
    vis = draw_header_and_legend(
        vis,
        instruction=instruction,
        traj_instruction=traj_instruction,
        gt_action=(dataset_action or traj_action),
        traj_action=traj_action,
        split_name=split_name,
    )
    return vis, sample_name, instruction


def make_grid_2x2(images_bgr: List[np.ndarray]) -> np.ndarray:
    if len(images_bgr) != 4:
        raise ValueError("need exactly 4 images for 2x2 grid")
    target_h = min(img.shape[0] for img in images_bgr)
    target_w = min(img.shape[1] for img in images_bgr)
    resized = [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA) for img in images_bgr]
    row1 = np.hstack([resized[0], resized[1]])
    row2 = np.hstack([resized[2], resized[3]])
    return np.vstack([row1, row2])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def guess_default_paths(script_file: str) -> Tuple[str, str]:
    script_dir = os.path.dirname(os.path.abspath(script_file))
    cands = [
        os.path.join(script_dir, "dist_all"),
        os.path.join(script_dir, "..", "paper_release_bundle", "data", "dist_all"),
    ]
    npz_dir = cands[0]
    for c in cands:
        if os.path.isdir(c):
            npz_dir = c
            break
    out_dir = os.path.join(os.path.dirname(npz_dir), "dataset_vis")
    return os.path.abspath(npz_dir), os.path.abspath(out_dir)


def main():
    default_npz_dir, default_out_dir = guess_default_paths(__file__)
    ap = argparse.ArgumentParser(description="Visualize dataset samples by split with instruction + regions.")
    ap.add_argument(
        "--npz_dir",
        type=str,
        default=default_npz_dir,
        help="Directory containing *.npz dataset files",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir,
        help="Output directory",
    )
    ap.add_argument("--traj_index", type=int, default=0, help="Which trajectory index to visualize per npz")
    ap.add_argument("--indices_0_99", type=str, default="", help="Optional 4 indices for split 0-99, e.g. 1,33,66,99")
    ap.add_argument("--indices_100_129", type=str, default="", help="Optional 4 indices for split 100-129")
    ap.add_argument("--indices_130_159", type=str, default="", help="Optional 4 indices for split 130-159")
    args = ap.parse_args()

    npz_dir = os.path.abspath(args.npz_dir)
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    by_idx = collect_npz_by_index(npz_dir)
    if not by_idx:
        raise RuntimeError(f"no npz files found under {npz_dir}")

    splits = [
        ("0-99", 0, 99),
        ("100-129", 100, 129),
        ("130-159", 130, 159),
    ]

    summary_lines = []
    manual_map = {
        "0-99": parse_indices_arg(args.indices_0_99, 0, 99),
        "100-129": parse_indices_arg(args.indices_100_129, 100, 129),
        "130-159": parse_indices_arg(args.indices_130_159, 130, 159),
    }

    for split_name, start, end in splits:
        split_dir = os.path.join(out_dir, split_name)
        ensure_dir(split_dir)
        target_indices = manual_map.get(split_name) or pick_four_indices(start, end)

        vis_images = []
        summary_lines.append(f"[{split_name}]")

        for idx in target_indices:
            if idx not in by_idx:
                summary_lines.append(f"  - idx={idx}: MISSING")
                continue
            npz_path = by_idx[idx]
            vis, sample_name, instruction = load_one_visual(npz_path, args.traj_index, split_name)

            out_name = f"{idx:03d}_{sample_name}.png"
            out_path = os.path.join(split_dir, out_name)
            cv2.imwrite(out_path, vis)

            vis_images.append(vis)
            summary_lines.append(f"  - idx={idx}: {out_name} | instruction={instruction}")

        if len(vis_images) == 4:
            grid = make_grid_2x2(vis_images)
            grid_path = os.path.join(out_dir, f"{split_name}_grid.png")
            cv2.imwrite(grid_path, grid)
            summary_lines.append(f"  - grid: {os.path.basename(grid_path)}")
        else:
            summary_lines.append("  - grid: skipped (not enough images)")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("Done.")
    print(f"npz_dir: {npz_dir}")
    print(f"out_dir: {out_dir}")
    print("Generated:")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
