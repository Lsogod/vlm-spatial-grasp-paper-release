# main_vision_only.py
# -*- coding: utf-8 -*-
import os
import io
import json
import uuid
import base64
import hashlib
import random
from typing import Optional

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, KMeans
from sklearn.metrics import silhouette_score

import torch
import torchvision.transforms as T

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import SAM
from kmeans_pytorch import kmeans

# ⚠️ 全局关闭严格确定性（warn_only 避免未来某些操作再报错）
torch.use_deterministic_algorithms(False, warn_only=True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ================= 环境与模型初始化 =================

print("Loading SAM model...")
sam_model = SAM("./model/sam2.1_l.pt")

print("Loading DINO model...")
dino_model = torch.hub.load('./model', 'dinov2_vitg14', source='local').eval().cuda()
print("Models loaded.")


# ================== 统一绘制函数：RGB 上画点+编号 ==================
def draw_numbered_points_rgb(
    img_rgb: np.ndarray,
    ids: np.ndarray,
    centers_xy: np.ndarray,
    label_scale: float = 0.0016,
    label_bg_alpha: float = 1.0,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb=(255, 0, 0),   # 默认红字（RGB）
    dot_color_rgb=(0, 0, 255),    # 默认蓝点（RGB）
    outline_color_rgb=(0, 0, 0),
    line_type=cv2.LINE_AA,
):
    """
    在 RGB 图像上绘制：点(带黑边) + 编号(白底/可透明 + 黑框 + 彩色字)。
    """
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be an RGB image (H,W,3).")

    out = img_rgb.copy()
    H, W = out.shape[:2]

    ids = np.asarray(ids).reshape(-1).astype(int)
    centers_xy = np.asarray(centers_xy).reshape(-1, 2).astype(float)
    if len(ids) != len(centers_xy):
        raise ValueError(f"ids length {len(ids)} != centers_xy length {len(centers_xy)}")

    # 自适应样式（按画布短边）
    short_side = min(H, W)
    font_scale = max(0.6, min(6.0, float(label_scale) * short_side))
    thickness  = max(1, int(0.0012 * short_side))
    pad        = max(4, int(0.0060 * short_side))
    radius     = max(3, int(0.0040 * short_side))

    for idx, (cx, cy) in zip(ids, centers_xy):
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if not (0 <= cx_i < W and 0 <= cy_i < H):
            continue

        # 点 + 黑边
        cv2.circle(out, (cx_i, cy_i), radius, dot_color_rgb, -1, lineType=line_type)
        cv2.circle(out, (cx_i, cy_i), radius + max(1, thickness // 2), outline_color_rgb, 1, lineType=line_type)

        # 文本框
        text = str(int(idx))
        (tw, th), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        box_w, box_h = tw + 2 * pad, th + 2 * pad

        # 标签默认放上方
        top_left = (cx_i - box_w // 2, cy_i - box_h // 2 - radius - pad)
        bottom_right = (top_left[0] + box_w, top_left[1] + box_h)

        # 防越界裁剪
        x1, y1 = max(0, top_left[0]), max(0, top_left[1])
        x2, y2 = min(W - 1, bottom_right[0]), min(H - 1, bottom_right[1])
        if x2 <= x1 or y2 <= y1:
            continue

        # 白底（可透明）
        if float(label_bg_alpha) >= 1.0:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1, lineType=line_type)
        else:
            roi = out[y1:y2, x1:x2].astype(np.float32)
            white = np.full_like(roi, 255, dtype=np.float32)
            out[y1:y2, x1:x2] = (float(label_bg_alpha) * white + (1 - float(label_bg_alpha)) * roi).astype(np.uint8)

        # 黑框 + 字
        cv2.rectangle(out, (x1, y1), (x2, y2), outline_color_rgb, max(1, thickness // 2), lineType=line_type)
        text_pos = (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2)
        cv2.putText(out, text, text_pos, font_face, font_scale, font_color_rgb, thickness, lineType=line_type)

    return out


def encode_rgb_to_data_url(img_rgb: np.ndarray, ext: str = ".png") -> str:
    """
    把 RGB uint8 图编码成 data url。
    ext: ".png" / ".jpg"
    """
    ext = ext.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        ext = ".png"
    mime = "image/png" if ext == ".png" else "image/jpeg"

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{base64_str}"


# ================== 自适应 K（每个 mask 的簇数） ==================
def choose_k_by_sqrt_area(n_patches: int, k_min: int, k_max: int, alpha: float) -> int:
    """
    基于 mask 覆盖 patch 数 n_patches 的自适应 K：
      K ~ alpha * sqrt(n_patches)
    """
    if n_patches <= 0:
        return 0
    k = int(round(alpha * (n_patches ** 0.5)))
    k = int(np.clip(k, k_min, min(k_max, n_patches)))
    return k


def feature_complexity(feats: np.ndarray) -> float:
    """
    feats: [N, C]，用简单的 mean(std) 作为复杂度指标（便宜、稳定）。
    返回值越大，说明内部差异越大。
    """
    if feats is None or feats.ndim != 2 or feats.shape[0] < 5:
        return 0.0
    return float(feats.std(axis=0).mean())


def choose_k_silhouette(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    sample_max: int,
    seed: int
) -> int:
    """
    用 silhouette score 在 [k_min, k_max] 中挑选 K（更贴合“多部件/多语义”的物体）。
    注意：计算更贵；建议只对很大的 mask 开启。
    """
    n = int(X.shape[0])
    if n <= 1:
        return 0
    k_min = int(max(2, k_min))
    k_max = int(min(k_max, n))
    if k_min > k_max:
        return k_max

    # 下采样加速
    if n > sample_max:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=sample_max, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    best_k, best_score = k_min, -1e9
    for k in range(k_min, min(k_max, Xs.shape[0]) + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        labels = km.fit_predict(Xs)
        if len(np.unique(labels)) < 2:
            continue
        sc = silhouette_score(Xs, labels, metric='euclidean')
        if sc > best_score:
            best_score = sc
            best_k = k
    return int(best_k)


# ================== 核心算法：SAM ==================
def run_sam_and_save_npz(
    image_path: str,
    sam_model,
    save_npz_path: str,
    min_area: int = 1
):
    """
    调用 SAM 对 image_path 做分割，输出并保存 masks, ids, centers_xy, image_size 到 npz。
    centers_xy 使用“最大内接圆中心”(distance transform argmax)，确保点在掩码内。
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # 运行 SAM
    results = sam_model(image_path)
    raw_masks = results[0].masks.data.cpu().numpy()  # [M, H, W]

    # 规范化为 uint8(0/1)
    masks = (raw_masks > 0).astype(np.uint8)

    kept_masks = []
    centers_xy = []
    ids = []

    for i, m in enumerate(masks):
        area = int(m.sum())
        if area < min_area:
            continue

        m_u8 = (m > 0).astype(np.uint8)
        if m_u8.sum() == 0:
            continue

        # ✅ 最大内接圆中心：离边界最远的 mask 内点
        dist = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(dist.argmax(), dist.shape)
        cx, cy = float(cx), float(cy)

        kept_masks.append(m_u8)
        centers_xy.append([cx, cy])
        ids.append(i)

    if len(kept_masks) == 0:
        kept_masks = np.zeros((0, H, W), dtype=np.uint8)
        centers_xy = np.zeros((0, 2), dtype=np.float32)
        ids = np.zeros((0,), dtype=np.int32)
    else:
        kept_masks = np.stack(kept_masks, axis=0).astype(np.uint8)
        centers_xy = np.array(centers_xy, dtype=np.float32)
        ids = np.array(ids, dtype=np.int32)

    os.makedirs(os.path.dirname(save_npz_path) or ".", exist_ok=True)
    np.savez_compressed(
        save_npz_path,
        masks=kept_masks,
        ids=ids,
        centers_xy=centers_xy,
        image_size=np.array([W, H], dtype=np.int32)
    )
    print(f"[SAM] Saved to {save_npz_path} | masks={kept_masks.shape} | W,H=({W},{H})")
    return save_npz_path


def draw_ids_and_centers_from_npz(
    image_path: str,
    npz_path: str,
    save_path: str = None,
    label_scale: float = 0.0016,
    label_bg_alpha: float = 1.0,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb=(255, 0, 0),   # 红字（RGB）
    dot_color_rgb=(0, 0, 255),    # 蓝点（RGB）
    return_base64: bool = True
):
    """
    从 npz 读取 ids 与 centers_xy 并在图像上绘制编号与中心点（统一 RGB 绘制，避免红蓝颠倒）。
    """
    # 读图（BGR -> RGB）
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # 读取 npz
    data = np.load(npz_path)
    ids = data.get("ids")
    centers_xy = data.get("centers_xy")
    npz_size = data.get("image_size", None)

    if ids is None or centers_xy is None:
        raise ValueError("npz 文件中缺少 'ids' 或 'centers_xy'")

    ids = ids.reshape(-1).astype(int)
    centers_xy = centers_xy.reshape(-1, 2).astype(float)

    # 尺寸匹配（鲁棒）
    if npz_size is not None:
        W_npz, H_npz = int(npz_size[0]), int(npz_size[1])
        if (W_npz, H_npz) != (W, H):
            sx, sy = W / W_npz, H / H_npz
            centers_xy *= np.array([sx, sy], dtype=np.float32)

    out_rgb = draw_numbered_points_rgb(
        img_rgb=img_rgb,
        ids=ids,
        centers_xy=centers_xy,
        label_scale=label_scale,
        label_bg_alpha=label_bg_alpha,
        font_face=font_face,
        font_color_rgb=font_color_rgb,
        dot_color_rgb=dot_color_rgb,
    )

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    if return_base64:
        ext = ".png"
        if save_path:
            sp = save_path.lower()
            if sp.endswith(".jpg") or sp.endswith(".jpeg"):
                ext = ".jpg"
            elif sp.endswith(".png"):
                ext = ".png"
        return encode_rgb_to_data_url(out_rgb, ext=ext)
    else:
        return out_rgb


# ================== 核心算法：DINO ==================
def visualize_dino_with_precomputed_masks(
    image_path: str,
    dino_model,
    config: dict,
    save_path: str = None,
    target_mask_indices=None,
    masks_npz_path: str = None,
    masks_array: np.ndarray = None,
    centers_xy: np.ndarray = None
):
    """
    使用“预先保存”的 SAM 掩码 npz（或直接传入 masks_array）做 DINO 特征提取与关键点生成。
    ✅ 改进点：
      - 每个 mask 的 K（簇数）自适应（sqrt_area / silhouette）
      - 可选：按特征复杂度对 K 做倍率修正
      - 绘制统一在原图尺寸上进行，避免字号/大小不一致
      - 绘制统一 RGB，避免红蓝反转
    """

    # ================== 工具：稳定随机性 & 派生种子 ==================
    def _deterministic_env(seed: int, strict: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def _seed_from_group(mask_indices, masks_array, base_seed=42):
        H, W = masks_array.shape[1], masks_array.shape[2]
        combined = np.zeros((H, W), dtype=np.uint8)
        for idx in sorted(mask_indices):
            combined |= (masks_array[idx] > 0).astype(np.uint8)
        h = hashlib.sha1(combined.tobytes()).hexdigest()
        return base_seed ^ int(h[:8], 16)

    # ================== 合并策略实现 ==================
    def _merge_meanshift_nearest(candidate_coords, bandwidth):
        if len(candidate_coords) == 0:
            return []
        pts = np.array(candidate_coords, dtype=np.float32)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pts)
        centers = ms.cluster_centers_
        if len(centers):
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            centers = centers[order]
        keep_indices = []
        for c in centers:
            d = np.linalg.norm(pts - c[None, :], axis=1)
            keep_indices.append(int(np.argmin(d)))
        sel_pts = pts[keep_indices]
        if len(sel_pts):
            order2 = np.lexsort((sel_pts[:, 0], sel_pts[:, 1]))
            keep_indices = [keep_indices[i] for i in order2]
        return keep_indices

    def _merge_meanshift_center(candidate_coords, bandwidth):
        if len(candidate_coords) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.array(candidate_coords, dtype=np.float32)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pts)
        labels = ms.labels_
        centers = []
        for l in sorted(np.unique(labels)):
            centers.append(pts[labels == l].mean(axis=0))
        centers = np.array(centers, dtype=np.float32)
        if len(centers):
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            centers = centers[order]
        return centers

    def _merge_by_distance_center(candidate_coords, min_dist_px):
        pts = np.array(candidate_coords, dtype=np.float32)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        order0 = np.lexsort((pts[:, 0], pts[:, 1]))
        pts = pts[order0]
        clusters = []
        for p in pts:
            attached = False
            for c in clusters:
                if np.linalg.norm(p - c["center"]) < float(min_dist_px):
                    c["members"].append(p)
                    c["center"] = np.mean(c["members"], axis=0)
                    attached = True
                    break
            if not attached:
                clusters.append({"members": [p], "center": p.copy()})
        centers = np.array([c["center"] for c in clusters], dtype=np.float32)
        if len(centers):
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            centers = centers[order]
        return centers

    # ================== 评分辅助 ==================
    def _normalize(x):
        x = np.asarray(x, np.float32)
        if x.size == 0:
            return x
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / (mx - mn + 1e-8)

    def _edge_map(gray):
        g1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        g2 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(g1 * g1 + g2 * g2)
        mag = _normalize(mag)
        return mag

    # ================== 全局随机种子 ==================
    base_seed = int(config.get('seed', 42))
    strict_det = bool(config.get('strict_deterministic', False))
    _deterministic_env(base_seed, strict=strict_det)

    # ================== 尺寸与特征 ==================
    patch_size = 14
    img = Image.open(image_path).convert('RGB')
    W_orig, H_orig = img.size

    patch_h = H_orig // patch_size
    patch_w = W_orig // patch_size
    H_new = patch_h * patch_size
    W_new = patch_w * patch_size

    # 载入掩码
    if masks_array is None:
        if masks_npz_path is None:
            raise ValueError("必须提供 masks_npz_path 或 masks_array 之一。")
        data = np.load(masks_npz_path)
        masks_array = data["masks"]
        if "centers_xy" in data:
            centers_xy = data["centers_xy"]
    else:
        masks_array = (masks_array > 0).astype(np.uint8)

    M = masks_array.shape[0]
    if M == 0:
        orig_full_rgb = np.array(img)
        return orig_full_rgb, [], encode_rgb_to_data_url(orig_full_rgb, ext=".png")

    # DINO 特征
    device = next(dino_model.parameters()).device
    transform = T.Compose([
        T.Resize((H_new, W_new)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino_model.forward_features(img_tensor)
    features = feats['x_norm_patchtokens'][0].cpu().numpy().reshape(patch_h, patch_w, -1)

    # aligned 底图（DINO 对齐尺寸）
    orig_rgb_aligned = np.array(img.resize((W_new, H_new), resample=Image.BILINEAR))
    local_pca_canvas = np.zeros((H_new, W_new, 3), dtype=np.uint8)

    # 目标分组
    if target_mask_indices is None:
        target_mask_indices = list(range(M))

    union_mask = np.zeros((H_orig, W_orig), dtype=bool)

    # 仅收集候选点
    all_points = []
    group_id_running = 0

    def _to_patch_bool(mask_bool_img):
        return cv2.resize(mask_bool_img.astype(np.uint8),
                          (patch_w, patch_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    # 标准化为二维分组
    groups = []
    for g in target_mask_indices:
        if isinstance(g, int):
            groups.append([g])
        else:
            groups.append(list(g))

    # ================== 主循环：每个分组 ==================
    for group in groups:
        hseed = _seed_from_group(group, masks_array, base_seed=base_seed)
        _deterministic_env(hseed, strict=strict_det)

        combined = np.zeros((H_orig, W_orig), dtype=bool)
        for idx in sorted(group):
            if 0 <= idx < M:
                combined |= (masks_array[idx] > 0)

        if not combined.any():
            group_id_running += 1
            continue

        union_mask |= combined

        mask_patch = _to_patch_bool(combined)
        n_patches = int(mask_patch.sum())
        ratio = float(mask_patch.mean())

        if n_patches < 1 or (ratio > config.get('max_mask_ratio', 0.5) and not config.get('visualize_large_masks', False)):
            group_id_running += 1
            continue

        region_pixels = np.argwhere(mask_patch)
        region_pixels = np.array(sorted(region_pixels.tolist(), key=lambda p: (p[0], p[1])))
        region_features = features[mask_patch]

        if region_pixels.shape[0] < 2:
            group_id_running += 1
            continue

        # ===== 可视化局部 PCA（可选）=====
        if config.get('semantic_visualize', True) and region_features.shape[0] >= 3:
            pca_vis = PCA(n_components=3, random_state=hseed)
            vis_feats = pca_vis.fit_transform(region_features)
            for c in range(3):
                col = vis_feats[:, c]
                rng = (col.max() - col.min() + 1e-8)
                vis_feats[:, c] = (col - col.min()) / rng
            vis_uint8 = (vis_feats * 255).astype(np.uint8)
            for (py, px), rgb in zip(region_pixels, vis_uint8):
                y0, x0 = py * patch_size, px * patch_size
                local_pca_canvas[y0:y0+patch_size, x0:x0+patch_size] = rgb

        # ===== 聚类输入特征（可选 local PCA）=====
        region_feats_for_kmeans = region_features
        if config.get("local_pca", True) and region_features.shape[0] >= 3:
            if config.get("use_sklearn_pca", True):
                pca3 = PCA(n_components=3, random_state=hseed)
                region_feats_for_kmeans = pca3.fit_transform(region_features)
            else:
                rt = torch.tensor(region_features, dtype=torch.float32)
                u, s, v = torch.pca_lowrank(rt, q=3, center=True)
                region_feats_for_kmeans = torch.matmul(rt, v[:, :3]).cpu().numpy()

        # ================== ✅ 自适应选择 K（簇数） ==================
        k_mode = str(config.get("k_mode", "sqrt_area")).lower()
        k_min = int(config.get("k_min", 3))
        k_max = int(config.get("k_max", 12))
        k_max = int(min(k_max, region_pixels.shape[0]))

        # 只对足够大的 mask 做 silhouette（避免太慢）
        silhouette_min_patches = int(config.get("silhouette_min_patches", 80))
        if k_mode == "silhouette" and n_patches >= silhouette_min_patches:
            k_try_min = int(config.get("k_try_min", 2))
            k_try_max = int(config.get("k_try_max", 10))
            k_try_max = int(min(k_try_max, k_max))
            sample_max = int(config.get("k_sample_max", 800))
            num_k = choose_k_silhouette(region_feats_for_kmeans, k_try_min, k_try_max, sample_max, hseed)
        else:
            alpha = float(config.get("k_sqrt_alpha", 0.6))
            num_k = choose_k_by_sqrt_area(n_patches, k_min, k_max, alpha)

            # 可选：按特征复杂度做倍率修正（纹理/多部件物体更大 K）
            if bool(config.get("use_complexity_k_boost", True)):
                comp = feature_complexity(region_feats_for_kmeans)
                # 把 comp 映射到一个温和倍率区间
                # comp 大 -> mult 大（上限 1.4），comp 小 -> mult 小（下限 0.8）
                mult = float(np.clip(0.8 + float(config.get("complexity_gain", 2.0)) * comp, 0.8, 1.4))
                num_k = int(np.clip(round(num_k * mult), k_min, k_max))

        # KMeans 至少要 2（否则 sklearn 会报错）
        if num_k < 2:
            group_id_running += 1
            continue

        num_k = int(min(num_k, region_pixels.shape[0]))
        use_sklearn_kmeans = bool(config.get("use_sklearn_kmeans", True))

        # ================== 聚类 ==================
        if use_sklearn_kmeans:
            km = KMeans(n_clusters=num_k, random_state=hseed, n_init='auto')
            cluster_ids_x = km.fit_predict(np.asarray(region_feats_for_kmeans))
            cluster_centers = km.cluster_centers_
        else:
            X_t = torch.tensor(
                region_feats_for_kmeans,
                dtype=torch.float32,
                device=torch.device(config.get('kmeans_device', 'cuda'))
            )
            cluster_ids_x, cluster_centers_t = kmeans(
                X=X_t,
                num_clusters=num_k,
                distance=config.get('kmeans_distance', 'cosine'),
                device=torch.device(config.get('kmeans_device', 'cuda'))
            )
            cluster_ids_x = cluster_ids_x.cpu().numpy()
            cluster_centers = cluster_centers_t.cpu().numpy()

        # 每个簇选代表点（离簇中心最近的 patch）
        candidate_coords = []
        for kk in range(num_k):
            members_idx = (cluster_ids_x == kk)
            members = np.asarray(region_feats_for_kmeans)[members_idx]
            members_px = np.asarray(region_pixels)[members_idx]
            if members.shape[0] == 0:
                continue

            dif = members - cluster_centers[kk][:members.shape[1]]
            ci = int(np.argmin(np.linalg.norm(dif, axis=1)))
            py, px = members_px[ci]
            cx = float(px * patch_size + patch_size // 2)
            cy = float(py * patch_size + patch_size // 2)
            candidate_coords.append([cx, cy])

        if len(candidate_coords):
            arr = np.array(candidate_coords, dtype=np.float32)
            idx_sort = np.lexsort((arr[:, 0], arr[:, 1]))
            candidate_coords = arr[idx_sort].tolist()

        # 合并策略
        merge_mode = str(config.get("merge_mode", "meanshift_center")).lower()
        bandwidth_px = float(config.get('min_dist_bt_keypoints', 0.06)) * H_new

        if merge_mode == "meanshift_nearest":
            keep_indices = _merge_meanshift_nearest(candidate_coords, bandwidth_px)
            merged_points = [candidate_coords[i] for i in keep_indices]
        elif merge_mode == "meanshift_center":
            centers = _merge_meanshift_center(candidate_coords, bandwidth_px)
            merged_points = centers.tolist()
        elif merge_mode == "distance_center":
            min_dist_px = float(config.get("merge_min_dist_px", bandwidth_px))
            centers = _merge_by_distance_center(candidate_coords, min_dist_px)
            merged_points = centers.tolist()
        else:
            raise ValueError(f"未知 merge_mode: {merge_mode}. 可选: meanshift_nearest | meanshift_center | distance_center")

        merged_points = sorted(merged_points, key=lambda p: (p[1], p[0]))
        for (cx, cy) in merged_points:
            all_points.append({"pt": (float(cx), float(cy)), "group": group_id_running})

        group_id_running += 1

    # ================== 评分 + Top-K ==================
    blend_mask = cv2.resize(union_mask.astype(np.uint8), (W_new, H_new),
                            interpolation=cv2.INTER_NEAREST).astype(bool)

    gray = cv2.cvtColor(orig_rgb_aligned, cv2.COLOR_RGB2GRAY)
    E = _edge_map(gray)

    mask_u8 = blend_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    B = np.zeros_like(mask_u8)
    if len(contours):
        cv2.drawContours(B, contours, -1, 1, thickness=1)
    D = cv2.distanceTransform((1 - B).astype(np.uint8), cv2.DIST_L2, 3)

    scores = []
    r = int(config.get('score_window_radius', 3))
    for it in all_points:
        cx, cy = it["pt"]
        cx, cy = int(round(cx)), int(round(cy))
        cx = int(np.clip(cx, 0, W_new - 1))
        cy = int(np.clip(cy, 0, H_new - 1))

        x1, x2 = max(0, cx - r), min(W_new - 1, cx + r)
        y1, y2 = max(0, cy - r), min(H_new - 1, cy + r)

        edge_local = float(E[y1:y2 + 1, x1:x2 + 1].mean())
        d_local = float(D[cy, cx])
        lvar_local = float(E[y1:y2 + 1, x1:x2 + 1].var())
        clus_score = 1.0

        scores.append({
            "pt": (cx, cy),
            "group": it["group"],
            "edge": edge_local,
            "bdist": d_local,
            "lvar": lvar_local,
            "clus": clus_score
        })

    if len(scores):
        w = config.get('score_weights', {'edge': 0.4, 'bdist': 0.25, 'lvar': 0.25, 'clus': 0.10})
        edge = _normalize([s['edge'] for s in scores])
        bdst = _normalize([s['bdist'] for s in scores])
        lvar = _normalize([s['lvar'] for s in scores])
        clus = _normalize([s['clus'] for s in scores])
        final_scores = (w.get('edge', 0.0) * edge +
                        w.get('bdist', 0.0) * bdst +
                        w.get('lvar', 0.0) * lvar +
                        w.get('clus', 0.0) * clus)
        for i, s in enumerate(scores):
            s["score"] = float(final_scores[i])
    else:
        return np.array(img), [], encode_rgb_to_data_url(np.array(img), ext=".png")

    per_mask_topk = int(config.get('per_mask_topk', 0))
    if per_mask_topk > 0 and len(scores):
        kept = []
        from collections import defaultdict
        bucket = defaultdict(list)
        for s in scores:
            bucket[s["group"]].append(s)
        for gid, arr in bucket.items():
            arr_sorted = sorted(arr, key=lambda x: -x["score"])
            kept.extend(arr_sorted[:per_mask_topk])
        scores = kept

    topk_after = int(config.get('topk_after_scoring', 0))
    if topk_after > 0 and len(scores) > topk_after:
        scores = sorted(scores, key=lambda x: -x["score"])[:topk_after]

    collected_points = [s["pt"] for s in scores]
    keypoints_xy = []
    if len(collected_points):
        collected_points = sorted(collected_points, key=lambda p: (p[1], p[0]))
        for i, (cx, cy) in enumerate(collected_points):
            keypoints_xy.append({"id": i, "x": int(round(cx)), "y": int(round(cy))})

    # ========== 语义融合（aligned 尺寸上）==========
    blended_aligned = orig_rgb_aligned.copy()
    if config.get('semantic_visualize', False):
        alpha = float(config.get('semantic_blend_alpha', 0.6))
        blended_aligned[blend_mask] = (
            alpha * local_pca_canvas[blend_mask] + (1.0 - alpha) * orig_rgb_aligned[blend_mask]
        ).astype(np.uint8)

        if config.get('mask_only', True):
            gray3 = np.mean(orig_rgb_aligned, axis=-1, keepdims=True).astype(np.uint8)
            blended_aligned[~blend_mask] = (
                0.3 * orig_rgb_aligned[~blend_mask] + 0.7 * gray3[~blend_mask]
            ).astype(np.uint8)

    # ================== ✅ 统一在“原图尺寸”上绘制 ==================
    orig_full_rgb = np.array(img)

    if config.get('semantic_visualize', False):
        canvas_rgb = cv2.resize(blended_aligned, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    else:
        canvas_rgb = orig_full_rgb.copy()

    if config.get('show_keypoints', True) and len(keypoints_xy):
        sx = float(W_orig) / float(W_new)
        sy = float(H_orig) / float(H_new)

        ids_kp = np.array([kp["id"] for kp in keypoints_xy], dtype=np.int32)
        centers_kp = np.array([[kp["x"] * sx, kp["y"] * sy] for kp in keypoints_xy], dtype=np.float32)

        canvas_rgb = draw_numbered_points_rgb(
            img_rgb=canvas_rgb,
            ids=ids_kp,
            centers_xy=centers_kp,
            label_scale=float(config.get('label_scale', 0.0016)),
            label_bg_alpha=float(config.get('label_bg_alpha', 1.0)),
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_color_rgb=tuple(config.get("font_color_rgb", (255, 0, 0))),  # 红字
            dot_color_rgb=tuple(config.get("dot_color_rgb", (0, 0, 255))),    # 蓝点
        )

    output_final = canvas_rgb

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(output_final, cv2.COLOR_RGB2BGR))

    base64_data_url = encode_rgb_to_data_url(output_final, ext=".png")

    if bool(config.get("debug_show_matplotlib", False)):
        plt.figure(figsize=(8, 6))
        plt.imshow(output_final)
        plt.title("DINO Keypoints (Original-size, Adaptive-K)")
        plt.axis('off')
        plt.tight_layout()

    return output_final, keypoints_xy, base64_data_url


# =============== DINO 配置（含自适应 K） ===============
DINO_CONFIG = {
    'seed': 42,
    'max_mask_ratio': 0.5,
    'visualize_large_masks': True,

    'use_sklearn_pca': True,
    'use_sklearn_kmeans': True,
    'strict_deterministic': True,

    'semantic_visualize': False,
    'semantic_blend_alpha': 0.6,
    'mask_only': True,

    'local_pca': True,
    'show_keypoints': True,
    'kmeans_distance': 'cosine',
    'kmeans_device': 'cuda',

    # ✅ 绘制样式统一（RGB）
    'label_scale': 0.0016,
    'label_bg_alpha': 1.0,
    'font_color_rgb': (255, 0, 0),  # 红字
    'dot_color_rgb': (0, 0, 255),   # 蓝点（想红点就改 (255,0,0)）

    # ✅ 自适应 K 设置
    # k_mode: 'sqrt_area'（快）或 'silhouette'（更贴合多部件，但更慢）
    'k_mode': 'sqrt_area',
    'k_min': 5,
    'k_max': 20,
    'k_sqrt_alpha': 0.8,

    # 复杂度倍率修正（推荐开）
    'use_complexity_k_boost': True,
    'complexity_gain': 2.0,

    # silhouette 仅对大 mask 启用（避免太慢）
    'silhouette_min_patches': 80,
    'k_try_min': 2,
    'k_try_max': 10,
    'k_sample_max': 800,

    # 近点合并策略
    'min_dist_bt_keypoints': 0.04,
    'merge_mode': 'meanshift_nearest',   # meanshift_nearest | meanshift_center | distance_center
    'merge_min_dist_px': 0,

    # 打分 + 筛选
    'score_window_radius': 3,
    'score_weights': {'edge': 0.50, 'bdist': 0.00, 'lvar': 0.35, 'clus': 0.10},
    'per_mask_topk': 5,
    'topk_after_scoring': 25,

    # Debug
    'debug_show_matplotlib': False,
}


# ================== FastAPI 应用 & 路由 ==================
app = FastAPI(title="Vision-Action FastAPI Pipeline (Separate APIs)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 工具：保存上传文件
async def save_upload_file(upload_file: UploadFile, dest_dir: str = "./tmp") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filename = upload_file.filename or f"{uuid.uuid4().hex}.png"
    ext = os.path.splitext(filename)[1] or ".png"
    file_path = os.path.join(dest_dir, f"{uuid.uuid4().hex}{ext}")

    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # ⚠️ 重置游标
    await upload_file.seek(0)
    return file_path


# 可视化相关工具函数
def draw_points_stage_rgb(
    img_rgb: np.ndarray,
    points_xy,                 # list[(x,y)] or np.ndarray [N,2]
    ids=None,                  # list[int] or None
    label_scale: float = 0.0016,
    label_bg_alpha: float = 1.0,
    dot_color_rgb=(0, 0, 255),     # 默认蓝点
    font_color_rgb=(255, 0, 0),    # 默认红字
    draw_labels: bool = True,
    title: str = None,
):
    """
    在 RGB 图像上画点（可带编号），可选在左上角画标题。
    """
    out = img_rgb.copy()
    H, W = out.shape[:2]

    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if ids is None:
        ids = list(range(len(pts)))
    ids = np.asarray(ids).reshape(-1).astype(int)

    short_side = min(H, W)
    font_scale = max(0.6, min(6.0, float(label_scale) * short_side))
    thickness  = max(1, int(0.0012 * short_side))
    pad        = max(4, int(0.0060 * short_side))
    radius     = max(3, int(0.0040 * short_side))

    for idx, (cx, cy) in zip(ids, pts):
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if not (0 <= cx_i < W and 0 <= cy_i < H):
            continue

        cv2.circle(out, (cx_i, cy_i), radius, dot_color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx_i, cy_i), radius + max(1, thickness // 2), (0, 0, 0), 1, lineType=cv2.LINE_AA)

        if not draw_labels:
            continue

        text = str(int(idx))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        box_w, box_h = tw + 2 * pad, th + 2 * pad
        tl = (cx_i - box_w // 2, cy_i - box_h // 2 - radius - pad)
        br = (tl[0] + box_w, tl[1] + box_h)

        x1, y1 = max(0, tl[0]), max(0, tl[1])
        x2, y2 = min(W - 1, br[0]), min(H - 1, br[1])
        if x2 <= x1 or y2 <= y1:
            continue

        if float(label_bg_alpha) >= 1.0:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        else:
            roi = out[y1:y2, x1:x2].astype(np.float32)
            white = np.full_like(roi, 255, dtype=np.float32)
            out[y1:y2, x1:x2] = (float(label_bg_alpha) * white + (1 - float(label_bg_alpha)) * roi).astype(np.uint8)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), max(1, thickness // 2), lineType=cv2.LINE_AA)
        text_pos = (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2)
        cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color_rgb, thickness, lineType=cv2.LINE_AA)

    if title:
        cv2.rectangle(out, (8, 8), (8 + 560, 8 + 34), (255, 255, 255), -1)
        cv2.rectangle(out, (8, 8), (8 + 560, 8 + 34), (0, 0, 0), 1)
        cv2.putText(out, title[:60], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return out
    
def visualize_dino_trace_with_precomputed_masks(
    image_path: str,
    dino_model,
    config: dict,
    masks_npz_path: str = None,     # 改为默认 None
    masks_array: np.ndarray = None, # 新增参数
    target_mask_indices=None,
    save_dir: str = None,
):
    """
    修改后的函数：支持直接传入 masks_array (内存数组)，无需读取 npz 文件。
    """
    patch_size = 14
    img = Image.open(image_path).convert('RGB')
    W_orig, H_orig = img.size
    patch_h = H_orig // patch_size
    patch_w = W_orig // patch_size
    H_new = patch_h * patch_size
    W_new = patch_w * patch_size

    # ==========================================
    # 修改开始：掩码加载逻辑兼容
    # ==========================================
    if masks_array is None:
        # 如果没传数组，就去读文件
        if masks_npz_path is None:
             raise ValueError("必须提供 masks_npz_path 或 masks_array 之一。")
        
        if not os.path.exists(masks_npz_path):
            raise FileNotFoundError(f"找不到文件: {masks_npz_path}")

        data = np.load(masks_npz_path)
        # 兼容不同的键名
        if "instance_masks" in data:
            masks_array = data["instance_masks"]
        elif "masks" in data:
            masks_array = data["masks"]
        else:
            raise KeyError(f"NPZ {masks_npz_path} 中缺少 'masks' 或 'instance_masks'")
    
    # 确保是 uint8 (0/1)
    masks_array = (masks_array > 0).astype(np.uint8)
    # ==========================================
    # 修改结束
    # ==========================================

    M = int(masks_array.shape[0])
    if target_mask_indices is None:
        target_mask_indices = list(range(M))

    # ... (以下代码保持原样，完全不需要动) ...
    # 辅助函数定义
    strict_det = bool(config.get('strict_deterministic', False))

    def _deterministic_env(seed: int, strict: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def _seed_from_group(mask_indices, masks_array, base_seed=42):
        H, W = masks_array.shape[1], masks_array.shape[2]
        combined = np.zeros((H, W), dtype=np.uint8)
        for idx in sorted(mask_indices):
            combined |= (masks_array[idx] > 0).astype(np.uint8)
        h = hashlib.sha1(combined.tobytes()).hexdigest()
        return base_seed ^ int(h[:8], 16)

    # DINO 特征提取
    device = next(dino_model.parameters()).device
    transform = T.Compose([
        T.Resize((H_new, W_new)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dino_model.forward_features(img_tensor)
    features = feats['x_norm_patchtokens'][0].cpu().numpy().reshape(patch_h, patch_w, -1)

    base_aligned = np.array(img.resize((W_new, H_new), resample=Image.BILINEAR)) 

    def _to_patch_bool(mask_bool_img):
        return cv2.resize(mask_bool_img.astype(np.uint8),
                          (patch_w, patch_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    # groups 规范化
    groups = []
    for g in target_mask_indices:
        if isinstance(g, int):
            groups.append([g])
        else:
            groups.append(list(g))

    # Trace 存储结构
    trace = {
        "meta": {
            "image_size": [W_orig, H_orig],
            "aligned_size": [W_new, H_new],
            "patch_size": patch_size,
            "k_mode": str(config.get("k_mode", "sqrt_area")),
            "merge_mode": str(config.get("merge_mode", "meanshift_center")),
            "per_mask_topk": int(config.get("per_mask_topk", 0)),
            "topk_after_scoring": int(config.get("topk_after_scoring", 0)),
        },
        "groups": [],
        "stages": {},
        "clusters": {
            "global_num_clusters": 0,
        }
    }

    candidates_all = []
    merged_all = []
    scored_all = []
    final_all = []

    union_mask = np.zeros((H_orig, W_orig), dtype=bool)

    # 簇 overlay 需要的 label map
    label_map_patch = np.full((patch_h, patch_w), -1, dtype=np.int32)
    global_cluster_offset = 0
    global_cluster_centers = []  
    global_cluster_center_ids = [] 

    base_seed = int(config.get("seed", 42))

    def _normalize(x):
        x = np.asarray(x, np.float32)
        if x.size == 0:
            return x
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / (mx - mn + 1e-8)

    def _edge_map(gray):
        g1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        g2 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(g1 * g1 + g2 * g2)
        return _normalize(mag)

    # 合并逻辑
    def _merge_meanshift_nearest(candidate_coords, bandwidth):
        if len(candidate_coords) == 0:
            return []
        pts = np.array(candidate_coords, dtype=np.float32)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pts)
        centers = ms.cluster_centers_
        if len(centers):
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            centers = centers[order]
        keep_indices = []
        for c in centers:
            d = np.linalg.norm(pts - c[None, :], axis=1)
            keep_indices.append(int(np.argmin(d)))
        sel_pts = pts[keep_indices]
        if len(sel_pts):
            order2 = np.lexsort((sel_pts[:, 0], sel_pts[:, 1]))
            keep_indices = [keep_indices[i] for i in order2]
        return keep_indices

    def _merge_meanshift_center(candidate_coords, bandwidth):
        if len(candidate_coords) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.array(candidate_coords, dtype=np.float32)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pts)
        labels = ms.labels_
        centers = []
        for l in sorted(np.unique(labels)):
            centers.append(pts[labels == l].mean(axis=0))
        centers = np.array(centers, dtype=np.float32)
        if len(centers):
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            centers = centers[order]
        return centers

    def _merge_by_distance_center(candidate_coords, min_dist_px):
        pts = np.array(candidate_coords, dtype=np.float32)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        order0 = np.lexsort((pts[:, 0], pts[:, 1]))
        pts = pts[order0]
        clusters = []
        for p in pts:
            attached = False
            for c in clusters:
                if np.linalg.norm(p - c["center"]) < float(min_dist_px):
                    c["members"].append(p)
                    c["center"] = np.mean(c["members"], axis=0)
                    attached = True
                    break
            if not attached:
                clusters.append({"members": [p], "center": p.copy()})
        centers = np.array([c["center"] for c in clusters], dtype=np.float32)
        if len(centers):
            order = np.lexsort((centers[:, 0], centers[:, 1]))
            centers = centers[order]
        return centers

    # 主循环：每个分组
    group_id_running = 0
    for group in groups:
        hseed = _seed_from_group(group, masks_array, base_seed=base_seed)
        _deterministic_env(hseed, strict=strict_det)

        combined = np.zeros((H_orig, W_orig), dtype=bool)
        for idx in sorted(group):
            if 0 <= idx < M:
                combined |= (masks_array[idx] > 0)

        info = {
            "group_id": group_id_running,
            "mask_indices": group,
            "skipped": False,
            "skip_reason": "",
            "n_patches": 0,
            "mask_ratio": 0.0,
            "chosen_k": 0,
            "k_mode": str(config.get("k_mode", "sqrt_area")),
        }

        if not combined.any():
            info["skipped"] = True
            info["skip_reason"] = "empty_combined"
            trace["groups"].append(info)
            group_id_running += 1
            continue

        union_mask |= combined

        mask_patch = _to_patch_bool(combined)
        n_patches = int(mask_patch.sum())
        ratio = float(mask_patch.mean())
        info["n_patches"] = n_patches
        info["mask_ratio"] = ratio

        if n_patches < 2:
            info["skipped"] = True
            info["skip_reason"] = "too_small"
            trace["groups"].append(info)
            group_id_running += 1
            continue

        if ratio > float(config.get("max_mask_ratio", 0.5)) and not bool(config.get("visualize_large_masks", False)):
            info["skipped"] = True
            info["skip_reason"] = "too_large_mask_ratio"
            trace["groups"].append(info)
            group_id_running += 1
            continue

        region_pixels = np.argwhere(mask_patch) 
        region_pixels = np.array(sorted(region_pixels.tolist(), key=lambda p: (p[0], p[1])), dtype=np.int32)
        region_features = features[mask_patch]

        if region_features.shape[0] < 2:
            info["skipped"] = True
            info["skip_reason"] = "no_features"
            trace["groups"].append(info)
            group_id_running += 1
            continue

        # local_pca
        region_feats_for_kmeans = region_features
        if bool(config.get("local_pca", True)) and region_features.shape[0] >= 3:
            if bool(config.get("use_sklearn_pca", True)):
                pca3 = PCA(n_components=3, random_state=hseed)
                region_feats_for_kmeans = pca3.fit_transform(region_features)
            else:
                rt = torch.tensor(region_features, dtype=torch.float32)
                u, s, v = torch.pca_lowrank(rt, q=3, center=True)
                region_feats_for_kmeans = torch.matmul(rt, v[:, :3]).cpu().numpy()

        # 选 K
        k_mode = str(config.get("k_mode", "sqrt_area")).lower()
        k_min = int(config.get("k_min", 3))
        k_max = int(config.get("k_max", 12))
        k_max = int(min(k_max, region_pixels.shape[0]))

        if k_mode == "silhouette" and n_patches >= int(config.get("silhouette_min_patches", 80)):
            num_k = choose_k_silhouette(
                region_feats_for_kmeans,
                int(config.get("k_try_min", 2)),
                int(min(config.get("k_try_max", 10), k_max)),
                int(config.get("k_sample_max", 800)),
                hseed
            )
        else:
            num_k = choose_k_by_sqrt_area(
                n_patches,
                k_min,
                k_max,
                float(config.get("k_sqrt_alpha", 0.6))
            )
            if bool(config.get("use_complexity_k_boost", True)):
                comp = feature_complexity(region_feats_for_kmeans)
                mult = float(np.clip(0.8 + float(config.get("complexity_gain", 2.0)) * comp, 0.8, 1.4))
                num_k = int(np.clip(round(num_k * mult), k_min, k_max))

        info["chosen_k"] = int(num_k)

        if num_k < 2:
            info["skipped"] = True
            info["skip_reason"] = "k<2"
            trace["groups"].append(info)
            group_id_running += 1
            continue

        num_k = int(min(num_k, region_pixels.shape[0]))

        # KMeans
        km = KMeans(n_clusters=num_k, random_state=hseed, n_init='auto')
        cluster_ids = km.fit_predict(np.asarray(region_feats_for_kmeans))
        cluster_centers = km.cluster_centers_

        # 写入 label_map_patch
        for (py, px), cid in zip(region_pixels, cluster_ids):
            label_map_patch[int(py), int(px)] = int(global_cluster_offset + int(cid))

        # 每个簇取代表点
        candidate_coords = []
        candidate_meta = []
        for cid in range(num_k):
            members_idx = (cluster_ids == cid)
            members = np.asarray(region_feats_for_kmeans)[members_idx]
            members_px = np.asarray(region_pixels)[members_idx]
            if members.shape[0] == 0:
                continue
            dif = members - cluster_centers[cid][:members.shape[1]]
            ci = int(np.argmin(np.linalg.norm(dif, axis=1)))
            py, px = members_px[ci]
            cx = float(px * patch_size + patch_size // 2)
            cy = float(py * patch_size + patch_size // 2)
            candidate_coords.append([cx, cy])
            candidate_meta.append({"cluster": int(cid)})

            global_cluster_centers.append([cx, cy])
            global_cluster_center_ids.append(int(global_cluster_offset + int(cid)))

        if len(candidate_coords):
            arr = np.array(candidate_coords, dtype=np.float32)
            idx_sort = np.lexsort((arr[:, 0], arr[:, 1]))
            candidate_coords = arr[idx_sort].tolist()
            candidate_meta = [candidate_meta[i] for i in idx_sort]

        for p, meta in zip(candidate_coords, candidate_meta):
            candidates_all.append({
                "pt": [float(p[0]), float(p[1])],
                "group": int(group_id_running),
                "cluster": int(meta["cluster"]),
                "global_cluster": int(global_cluster_offset + int(meta["cluster"]))
            })

        # 合并
        bandwidth_px = float(config.get("min_dist_bt_keypoints", 0.06)) * H_new
        merge_mode = str(config.get("merge_mode", "meanshift_center")).lower()

        if merge_mode == "meanshift_center":
            merged = _merge_meanshift_center(candidate_coords, bandwidth_px).tolist()
        elif merge_mode == "meanshift_nearest":
            keep_indices = _merge_meanshift_nearest(candidate_coords, bandwidth_px)
            merged = [candidate_coords[i] for i in keep_indices]
        elif merge_mode == "distance_center":
            min_dist_px = float(config.get("merge_min_dist_px", bandwidth_px))
            centers = _merge_by_distance_center(candidate_coords, min_dist_px)
            merged = centers.tolist()
        else:
            merged = _merge_meanshift_center(candidate_coords, bandwidth_px).tolist()

        merged = sorted(merged, key=lambda p: (p[1], p[0]))
        for p in merged:
            merged_all.append({
                "pt": [float(p[0]), float(p[1])],
                "group": int(group_id_running)
            })

        trace["groups"].append(info)
        global_cluster_offset += int(num_k)
        group_id_running += 1

    trace["clusters"]["global_num_clusters"] = int(global_cluster_offset)

    # 打分
    blend_mask = cv2.resize(union_mask.astype(np.uint8), (W_new, H_new),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    gray = cv2.cvtColor(base_aligned, cv2.COLOR_RGB2GRAY)
    E = _edge_map(gray)

    mask_u8 = blend_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    B = np.zeros_like(mask_u8)
    if len(contours):
        cv2.drawContours(B, contours, -1, 1, thickness=1)
    D = cv2.distanceTransform((1 - B).astype(np.uint8), cv2.DIST_L2, 3)

    scores = []
    r = int(config.get("score_window_radius", 3))
    for it in merged_all:
        cx, cy = it["pt"]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        cx_i = int(np.clip(cx_i, 0, W_new - 1))
        cy_i = int(np.clip(cy_i, 0, H_new - 1))

        x1, x2 = max(0, cx_i - r), min(W_new - 1, cx_i + r)
        y1, y2 = max(0, cy_i - r), min(H_new - 1, cy_i + r)

        edge_local = float(E[y1:y2 + 1, x1:x2 + 1].mean())
        d_local = float(D[cy_i, cx_i])
        lvar_local = float(E[y1:y2 + 1, x1:x2 + 1].var())
        scores.append({
            "pt": [float(cx_i), float(cy_i)],
            "group": int(it["group"]),
            "edge": edge_local,
            "bdist": d_local,
            "lvar": lvar_local,
            "clus": 1.0
        })

    if len(scores):
        w = config.get("score_weights", {'edge': 0.4, 'bdist': 0.25, 'lvar': 0.25, 'clus': 0.10})
        edge = _normalize([s['edge'] for s in scores])
        bdst = _normalize([s['bdist'] for s in scores])
        lvar = _normalize([s['lvar'] for s in scores])
        clus = _normalize([s['clus'] for s in scores])
        final_scores = (w.get('edge', 0.0) * edge +
                        w.get('bdist', 0.0) * bdst +
                        w.get('lvar', 0.0) * lvar +
                        w.get('clus', 0.0) * clus)
        for i, s in enumerate(scores):
            s["score"] = float(final_scores[i])

    scored_all = scores.copy()

    # per_mask_topk / topk_after_scoring
    per_mask_topk = int(config.get("per_mask_topk", 0))
    if per_mask_topk > 0 and len(scored_all):
        from collections import defaultdict
        bucket = defaultdict(list)
        for s in scored_all:
            bucket[s["group"]].append(s)
        kept = []
        for gid, arr in bucket.items():
            arr_sorted = sorted(arr, key=lambda x: -x["score"])
            kept.extend(arr_sorted[:per_mask_topk])
        scored_after_permask = kept
    else:
        scored_after_permask = scored_all

    topk_after = int(config.get("topk_after_scoring", 0))
    if topk_after > 0 and len(scored_after_permask) > topk_after:
        final_sel = sorted(scored_after_permask, key=lambda x: -x["score"])[:topk_after]
    else:
        final_sel = scored_after_permask

    final_pts = [s["pt"] for s in final_sel]
    final_pts = sorted(final_pts, key=lambda p: (p[1], p[0]))
    final_all = [{"id": i, "pt": [float(p[0]), float(p[1])]} for i, p in enumerate(final_pts)]

    trace["stages"]["candidates"] = candidates_all
    trace["stages"]["merged"] = merged_all
    trace["stages"]["scored"] = scored_all
    trace["stages"]["after_per_mask_topk"] = scored_after_permask
    trace["stages"]["final"] = final_all

    # 生成可视化图
    def _to_orig(img_aligned_rgb):
        return cv2.resize(img_aligned_rgb, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    # 1) candidates
    cand_pts = [c["pt"] for c in candidates_all]
    img_candidates = draw_points_stage_rgb(
        base_aligned, cand_pts,
        ids=list(range(len(cand_pts))),
        label_scale=float(config.get("label_scale", 0.0016)),
        label_bg_alpha=float(config.get("label_bg_alpha", 1.0)),
        dot_color_rgb=(0, 255, 255),
        font_color_rgb=(255, 0, 0),
        draw_labels=False,
        title="Stage: candidates (per-cluster representative, pre-merge)"
    )

    # 2) merged
    m_pts = [m["pt"] for m in merged_all]
    img_merged = draw_points_stage_rgb(
        base_aligned, m_pts,
        ids=list(range(len(m_pts))),
        label_scale=float(config.get("label_scale", 0.0016)),
        label_bg_alpha=float(config.get("label_bg_alpha", 1.0)),
        dot_color_rgb=(0, 0, 255),
        font_color_rgb=(255, 0, 0),
        draw_labels=False,
        title="Stage: merged (after merge_mode)"
    )

    # 3) scored
    s_pts = [s["pt"] for s in scored_all]
    img_scored = draw_points_stage_rgb(
        base_aligned, s_pts,
        ids=list(range(len(s_pts))),
        label_scale=float(config.get("label_scale", 0.0016)),
        label_bg_alpha=float(config.get("label_bg_alpha", 1.0)),
        dot_color_rgb=(255, 0, 255),
        font_color_rgb=(255, 0, 0),
        draw_labels=False,
        title="Stage: scored (before top-k filters)"
    )

    # 4) after_per_mask_topk
    ap_pts = [s["pt"] for s in scored_after_permask]
    img_after_permask = draw_points_stage_rgb(
        base_aligned, ap_pts,
        ids=list(range(len(ap_pts))),
        label_scale=float(config.get("label_scale", 0.0016)),
        label_bg_alpha=float(config.get("label_bg_alpha", 1.0)),
        dot_color_rgb=(0, 255, 0),
        font_color_rgb=(255, 0, 0),
        draw_labels=False,
        title=f"Stage: after per_mask_topk={per_mask_topk}"
    )

    # 5) final
    f_pts = [x["pt"] for x in final_all]
    f_ids = [x["id"] for x in final_all]
    img_final = draw_points_stage_rgb(
        base_aligned, f_pts,
        ids=f_ids,
        label_scale=float(config.get("label_scale", 0.0016)),
        label_bg_alpha=float(config.get("label_bg_alpha", 1.0)),
        dot_color_rgb=(255, 0, 0),
        font_color_rgb=(255, 0, 0),
        draw_labels=True,
        title="Stage: final selected keypoints (numbered)"
    )

    # 簇 overlay
    cluster_overlay_aligned = base_aligned.copy()
    alpha = float(config.get("cluster_overlay_alpha", 0.45))

    label_map_aligned = cv2.resize(label_map_patch.astype(np.int32), (W_new, H_new), interpolation=cv2.INTER_NEAREST)
    valid = (label_map_aligned >= 0)

    colors = _make_cluster_colors(global_cluster_offset, seed=int(config.get("seed", 42)) + 123)
    overlay = np.zeros_like(cluster_overlay_aligned, dtype=np.uint8)
    if global_cluster_offset > 0:
        idx = label_map_aligned.copy()
        idx[~valid] = 0
        overlay = colors[idx]

    cluster_overlay_aligned[valid] = (alpha * overlay[valid] + (1.0 - alpha) * cluster_overlay_aligned[valid]).astype(np.uint8)

    cluster_overlay_with_centers_aligned = cluster_overlay_aligned.copy()
    if len(global_cluster_centers) > 0:
        cluster_overlay_with_centers_aligned = draw_points_stage_rgb(
            cluster_overlay_with_centers_aligned,
            points_xy=np.array(global_cluster_centers, dtype=np.float32),
            ids=np.array(global_cluster_center_ids, dtype=np.int32),
            label_scale=float(config.get("label_scale", 0.0016)),
            label_bg_alpha=float(config.get("label_bg_alpha", 1.0)),
            dot_color_rgb=(255, 0, 0),
            font_color_rgb=(255, 0, 0),
            draw_labels=True,
            title="Stage: cluster_overlay (global cluster id numbered)"
        )

    images = {
        "cluster_overlay": encode_rgb_to_data_url(_to_orig(cluster_overlay_aligned), ext=".png"),
        "cluster_overlay_with_centers": encode_rgb_to_data_url(_to_orig(cluster_overlay_with_centers_aligned), ext=".png"),
        "candidates": encode_rgb_to_data_url(_to_orig(img_candidates), ext=".png"),
        "merged": encode_rgb_to_data_url(_to_orig(img_merged), ext=".png"),
        "scored": encode_rgb_to_data_url(_to_orig(img_scored), ext=".png"),
        "after_per_mask_topk": encode_rgb_to_data_url(_to_orig(img_after_permask), ext=".png"),
        "final": encode_rgb_to_data_url(_to_orig(img_final), ext=".png"),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "trace_cluster_overlay.png"), cv2.cvtColor(_to_orig(cluster_overlay_aligned), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "trace_cluster_overlay_with_centers.png"), cv2.cvtColor(_to_orig(cluster_overlay_with_centers_aligned), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "trace_candidates.png"), cv2.cvtColor(_to_orig(img_candidates), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "trace_merged.png"), cv2.cvtColor(_to_orig(img_merged), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "trace_scored.png"), cv2.cvtColor(_to_orig(img_scored), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "trace_after_permask.png"), cv2.cvtColor(_to_orig(img_after_permask), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "trace_final.png"), cv2.cvtColor(_to_orig(img_final), cv2.COLOR_RGB2BGR))

    return images, trace

def _make_cluster_colors(num_k: int, seed: int = 0):
    """
    生成 num_k 个稳定 RGB 颜色（确定性：同 seed+num_k 输出一致）
    """
    rng = np.random.RandomState(seed)
    colors = rng.randint(0, 256, size=(max(num_k, 1), 3), dtype=np.uint8)
    colors = np.clip(colors, 50, 255).astype(np.uint8)  # 避免过暗
    return colors


def draw_points_stage_rgb(
    img_rgb: np.ndarray,
    points_xy,                 # list[(x,y)] or np.ndarray [N,2]
    ids=None,                  # list[int] or None
    label_scale: float = 0.0016,
    label_bg_alpha: float = 1.0,
    dot_color_rgb=(0, 0, 255),     # 默认蓝点（RGB）
    font_color_rgb=(255, 0, 0),    # 默认红字（RGB）
    draw_labels: bool = True,
    title: str = None,
):
    """
    在 RGB 图像上画点（可带编号），可选在左上角画标题。
    """
    out = img_rgb.copy()
    H, W = out.shape[:2]

    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if ids is None:
        ids = list(range(len(pts)))
    ids = np.asarray(ids).reshape(-1).astype(int)

    short_side = min(H, W)
    font_scale = max(0.6, min(6.0, float(label_scale) * short_side))
    thickness  = max(1, int(0.0012 * short_side))
    pad        = max(4, int(0.0060 * short_side))
    radius     = max(3, int(0.0040 * short_side))

    for idx, (cx, cy) in zip(ids, pts):
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if not (0 <= cx_i < W and 0 <= cy_i < H):
            continue

        cv2.circle(out, (cx_i, cy_i), radius, dot_color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx_i, cy_i), radius + max(1, thickness // 2), (0, 0, 0), 1, lineType=cv2.LINE_AA)

        if not draw_labels:
            continue

        text = str(int(idx))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        box_w, box_h = tw + 2 * pad, th + 2 * pad
        tl = (cx_i - box_w // 2, cy_i - box_h // 2 - radius - pad)
        br = (tl[0] + box_w, tl[1] + box_h)

        x1, y1 = max(0, tl[0]), max(0, tl[1])
        x2, y2 = min(W - 1, br[0]), min(H - 1, br[1])
        if x2 <= x1 or y2 <= y1:
            continue

        if float(label_bg_alpha) >= 1.0:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        else:
            roi = out[y1:y2, x1:x2].astype(np.float32)
            white = np.full_like(roi, 255, dtype=np.float32)
            out[y1:y2, x1:x2] = (float(label_bg_alpha) * white + (1 - float(label_bg_alpha)) * roi).astype(np.uint8)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), max(1, thickness // 2), lineType=cv2.LINE_AA)
        text_pos = (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2)
        cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color_rgb, thickness, lineType=cv2.LINE_AA)

    if title:
        cv2.rectangle(out, (8, 8), (8 + 720, 8 + 34), (255, 255, 255), -1)
        cv2.rectangle(out, (8, 8), (8 + 720, 8 + 34), (0, 0, 0), 1)
        cv2.putText(out, title[:80], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return out

# ================== 1. 独立的 SAM 接口 ==================
@app.post("/vision/sam")
async def api_run_sam(
    image: UploadFile = File(...),
    min_area: int = Form(1)
):
    """
    独立运行 SAM 模型。
    返回：npz 路径 + 中心点可视化 data url
    """
    try:
        img_path = await save_upload_file(image)

        npz_path = os.path.join(os.path.dirname(img_path), f"{os.path.basename(img_path)}_sam.npz")
        run_sam_and_save_npz(img_path, sam_model, npz_path, min_area=min_area)

        vis_path = os.path.join(os.path.dirname(img_path), f"{os.path.basename(img_path)}_centers.png")
        centers_vis_data_url = draw_ids_and_centers_from_npz(
            image_path=img_path,
            npz_path=npz_path,
            save_path=vis_path,
            return_base64=True,
            label_scale=0.0016,
            label_bg_alpha=1.0,
            font_face=cv2.FONT_HERSHEY_SIMPLEX,
            font_color_rgb=(255, 0, 0),
            dot_color_rgb=(0, 0, 255),
        )

        return {
            "status": "success",
            "image_path": img_path,
            "npz_path": npz_path,
            "centers_vis_base64": centers_vis_data_url
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================== 2. 独立的 DINO 接口 ==================
@app.post("/vision/dino")
async def api_run_dino(
    image: UploadFile = File(...),
    npz_path: str = Form(..., description="服务端已存在的npz文件路径，通常由 /vision/sam 返回"),
    target_ids: Optional[str] = Form(None, description="可选：逗号分隔的ID列表")
):
    """
    独立运行 DINO 模型 + 关键点提取（自适应 K）。
    需要先调用 SAM 接口获取 npz_path。
    """
    try:
        img_path = await save_upload_file(image)

        target_mask_indices = None
        if target_ids:
            try:
                stripped = target_ids.strip()
                if stripped.startswith("["):
                    target_mask_indices = json.loads(stripped)
                else:
                    target_mask_indices = [int(x) for x in stripped.split(",") if x.strip()]
            except Exception:
                print("解析 target_ids 失败，将使用全部掩码")
                target_mask_indices = None

        if not os.path.exists(npz_path):
            return JSONResponse(status_code=400, content={"error": f"npz file not found: {npz_path}"})

        vis_save_path = os.path.join(os.path.dirname(img_path), f"{os.path.basename(img_path)}_dino_vis.png")
        _, keypoints, base64_img = visualize_dino_with_precomputed_masks(
            image_path=img_path,
            dino_model=dino_model,
            config=DINO_CONFIG,
            save_path=vis_save_path,
            target_mask_indices=target_mask_indices,
            masks_npz_path=npz_path
        )

        return {
            "status": "success",
            "keypoints": keypoints,
            "dino_vis_base64": base64_img
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/vision/dino_trace")
async def api_run_dino_trace(
    # 1. 接收图片文件 (用于DINO提取特征)
    image: UploadFile = File(...),
    # 2. 接收 NPZ 文件 (用于直接读取掩码)
    npz_file: UploadFile = File(..., description="客户端上传的 .npz 文件"), 
    # 3. 接收可选的 ID 列表
    target_ids: Optional[str] = Form(None, description="可选：逗号分隔的ID列表")
):
    """
    接收上传的图片和 .npz 文件，直接在内存中解析掩码并进行 DINO 可视化。
    """
    try:
        # --- 步骤 A: 处理图片 ---
        # 由于核心函数 visualize_dino_trace... 目前设计为接收文件路径
        # 我们先把上传的图片保存为临时文件
        img_path = await save_upload_file(image)

        # --- 步骤 B: 处理 NPZ 文件 ---
        # 直接读取上传文件的字节流，无需保存到硬盘
        content = await npz_file.read()
        
        # 使用 io.BytesIO 将字节流包装成文件对象，供 np.load 读取
        with io.BytesIO(content) as f:
            try:
                data = np.load(f)
                
                # 兼容不同的键名 (您的脚本生成的是 'instance_masks'，旧版是 'masks')
                if 'instance_masks' in data:
                    masks_array = data['instance_masks']
                elif 'masks' in data:
                    masks_array = data['masks']
                else:
                    return JSONResponse(status_code=400, content={"error": "NPZ file must contain 'instance_masks' or 'masks'"})
                
                # 确保是 uint8 类型 (0/1)
                masks_array = (masks_array > 0).astype(np.uint8)
                
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid NPZ file: {str(e)}"})

        # --- 步骤 C: 解析 target_ids ---
        target_mask_indices = None
        if target_ids:
            try:
                stripped = target_ids.strip()
                if stripped.startswith("["):
                    target_mask_indices = json.loads(stripped)
                else:
                    target_mask_indices = [int(x) for x in stripped.split(",") if x.strip()]
            except Exception:
                print("解析 target_ids 失败，将使用全部掩码")
                target_mask_indices = None

        # --- 步骤 D: 调用核心绘图逻辑 ---
        # 这里的关键是：直接传入 masks_array，不传 masks_npz_path
        save_dir = os.path.join(os.path.dirname(img_path), f"{os.path.splitext(os.path.basename(img_path))[0]}_trace")
        
        images, trace = visualize_dino_trace_with_precomputed_masks(
            image_path=img_path,        # 传入刚保存的临时图片路径
            dino_model=dino_model,
            config=DINO_CONFIG,
            masks_npz_path=None,        # 设为 None
            masks_array=masks_array,    # 传入内存中解析好的掩码数组
            target_mask_indices=target_mask_indices,
            save_dir=save_dir
        )

        # 清理临时图片 (可选)
        # if os.path.exists(img_path):
        #    os.remove(img_path)

        return {
            "status": "success",
            "images": images,
            "trace": trace
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
