import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

import json
import re
import base64
import textwrap
import queue
import time
import io
import os
from collections import OrderedDict
from typing import List, Dict, Literal, Tuple
from PIL import Image, ImageDraw, ImageFont

ReturnType = Literal["point", "bbox", "auto"]
from scipy.io.wavfile import write

from openai import OpenAI  # 导入OpenAI客户端

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# 基础工具函数 
def rel_box_to_px(bbox_rel, w, h):
    """将 [0,1000] 相对坐标的 bbox 换算为像素 bbox"""
    x1 = int(round(bbox_rel[0] / 1000.0 * w))
    y1 = int(round(bbox_rel[1] / 1000.0 * h))
    x2 = int(round(bbox_rel[2] / 1000.0 * w))
    y2 = int(round(bbox_rel[3] / 1000.0 * h))
    # 规范顺序与裁剪
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
    return [x1, y1, x2, y2]

def maybe_to_pixel_bbox(bbox, img_bgr):
    """
    如果 bbox 看起来是相对坐标（<=1000），就换算成像素；
    否则视为已经是像素坐标，直接返回规范化结果。
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    # 判定：全部在 [0,1000] 内 → 当作相对坐标
    if 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000:
        bb_px = rel_box_to_px([x1, y1, x2, y2], w, h)
    else:
        # 视为像素坐标；做一次顺序规范与裁剪
        x1, x2 = sorted([int(round(x1)), int(round(x2))])
        y1, y2 = sorted([int(round(y1)), int(round(y2))])
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        bb_px = [x1, y1, x2, y2]
    return bb_px

def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64



# 多模态模型调用
def generate_robot_actions(user_command, image_input=None):
    # 1) 保存输入图（若有）
    saved_image_path = None
    if image_input is not None:
        os.makedirs("logs/input_images", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        saved_image_path = os.path.join("logs/input_images", f"input_to_model_{ts}.jpg")
        cv2.imwrite(saved_image_path, image_input)
        print(f"🖼️ 已保存输入图像: {saved_image_path}")

    # 2) 初始化客户端（将占位符替换为你的密钥与基址）
    client = OpenAI(
        api_key=(os.getenv("MODELSCOPE_API_KEY") or "").strip(),
        base_url=(os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1") or "").rstrip("/")
    )

    # 3) 完整的 system prompt
    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知与指令理解能力。请严格遵循以下流程并输出规定格式：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点与右下角点，像素坐标）及其类别名称。

    【指令解析阶段】
    2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体（仅选择一个最合理的目标）。

    【响应生成阶段（必须严格遵守以下输出格式）】
    3. 输出由两部分组成：
       - 第一部分：仅包含被选中物体的自然语言说明（可以俏皮可爱地回应理由，但不要提及未被选中的物体）。
       - 紧接着从下一行开始，输出一个“标准 JSON 对象”，格式如下（不要添加任何多余文字、注释或标识）：
         {
           "name": "物体名称",
           "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
         }

    【注意事项】
    - 自然语言说明与 JSON 之间不得包含额外文本或空行。
    - JSON 对象只能包含 "name" 与 "bbox" 两个键；"bbox" 是 4 个整数像素坐标，含义分别为 [x1, y1, x2, y2]。
    - 严禁输出除上述 JSON 之外的任何解释、注释、Markdown、代码块标识或说明。
    """)

    # 4) 组织消息
    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    if image_input is not None:
        base64_img = encode_np_array(image_input)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    try:
        # 5) 调用多模态模型
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-VL-235B-A22B-Instruct",
            messages=messages,
            temperature=0.1,  # 低温度有助于稳定结构化输出
        )

        # 6) 解析输出：上半段自然语言 + 下半段 JSON
        content = completion.choices[0].message.content
        print("原始响应：", content)

        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                coord = json.loads(json_str)
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                coord = {}
            natural_response = content[:match.start()].strip()
        else:
            natural_response = content.strip()
            coord = {}

        # 7) 若拿到 bbox，则在保存的输入图上绘制并另存
        annotated_image_path = None
        bbox = coord.get("bbox") if isinstance(coord, dict) else None
        print("解析到的坐标：", coord)
        print(type(coord))
        if saved_image_path and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            # 👇 新增：相对→像素换算（自动判断）
            bbox_px = maybe_to_pixel_bbox(bbox, image_input)
            print(f"🧭 绘制用像素 bbox：{bbox_px}（图像尺寸 W={image_input.shape[1]}, H={image_input.shape[0]}）")
            annotated_image_path = draw_bbox_on_image(saved_image_path, bbox_px)

        return {
            "response": natural_response,
            "coordinates": coord,
            "saved_image_path": saved_image_path,
            "annotated_image_path": annotated_image_path
        }

    except Exception as e:
        print(f"请求失败：{e}")
        return {
            "response": "处理失败",
            "coordinates": {},
            "saved_image_path": saved_image_path,
            "annotated_image_path": None
        }
    
def draw_bbox_on_image(image_path, bbox, output_path=None, color=(0, 255, 0), thickness=3):
    """
    在图像上绘制大模型返回的边界框并保存新图片。

    参数：
        image_path (str): 原图路径
        bbox (list[int]): [x1, y1, x2, y2] 坐标
        output_path (str): 输出文件路径（可选）
        color (tuple): 边框颜色 (B, G, R)
        thickness (int): 边框线条粗细
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        print("⚠️ 无效的bbox格式，应为 [x1, y1, x2, y2]")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"annotated_{timestamp}.jpg"

    cv2.imwrite(output_path, image)
    print(f"✅ 已保存绘制边框的图片: {output_path}")

    return output_path

# SAM 分割相关 
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)

def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 状态警告：", status)
    q.put(bytes(indata))

def _bbox_center(bbox):
    # bbox: [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _point_in_mask(mask, point_xy):
    H, W = mask.shape[:2]
    x, y = int(round(point_xy[0])), int(round(point_xy[1]))
    if x < 0 or x >= W or y < 0 or y >= H:
        return False
    return mask[y, x] > 0

def _make_circular_mask(shape_hw, center_xy, radius_px):
    H, W = shape_hw[:2]
    roi = np.zeros((H, W), dtype=np.uint8)
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(roi, (cx, cy), int(round(radius_px)), 255, thickness=-1)
    return roi

def _make_annulus_mask(shape_hw, center_xy, r_inner_px, r_outer_px):
    H, W = shape_hw[:2]
    roi = np.zeros((H, W), dtype=np.uint8)
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(roi, (cx, cy), int(round(r_outer_px)), 255, thickness=-1)
    cv2.circle(roi, (cx, cy), int(round(r_inner_px)), 0, thickness=-1)
    return roi

def segment_and_crop_mask(
    image_bgr,
    *,
    point_xy,                 # 必须传入 (x, y)
    radius_px=50,             # 圆形半径
    annulus=None,             # 可选：(r_inner, r_outer)
    depth=None,               # 可选：深度图 (H, W)
    depth_max=2.0,            # 深度抑制阈值（米）
    save_path='mask_final.png'
):
    """
    生成局部掩码：(圆/环 ROI) ∩ (SAM 根据该点分割得到的物体掩码) ∩ (可选深度过滤)。
    返回：final_mask (uint8, HxW, 0/255)
    """
    H, W = image_bgr.shape[:2]
    px, py = int(round(point_xy[0])), int(round(point_xy[1]))
    assert 0 <= px < W and 0 <= py < H, "point_xy 超出图像范围"

    # --- 1) 深度抑制（可选） ---
    mask_depth = np.ones((H, W), dtype=np.uint8) * 255
    if depth is not None:
        d = np.asarray(depth)
        if d.shape[:2] != (H, W):
            # 深度图尺寸对齐
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_depth = ((d < float(depth_max))).astype(np.uint8) * 255

    # --- 2) 生成局部 ROI（圆或环） ---
    if annulus is not None:
        r_in, r_out = annulus
        roi = _make_annulus_mask((H, W), (px, py), r_in, r_out)  # 0/255
    else:
        roi = _make_circular_mask((H, W), (px, py), radius_px)   # 0/255

    # --- 3) SAM 点提示分割：获取“包含该点”的物体掩码 ---
    # 依赖你已有的 choose_model()
    predictor = choose_model()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        # 优先使用点提示
        results = predictor(image_rgb, points=[[px, py]])
    except Exception as e:
        print(f"[提示] 点提示失败，回退到整图自动分割：{e}")
        results = predictor(image_rgb)

    obj_mask = np.zeros((H, W), dtype=np.uint8)  # 默认全黑
    if results and results[0].masks is not None and len(results[0].masks.data) > 0:
        # 在所有候选掩码中挑选“包含该点”的，若多于一个取面积更大者
        best_area = -1
        for m_t in results[0].masks.data:
            m = m_t.cpu().numpy()
            # 二值化到 0/255
            if m.dtype.kind == "f":
                m = (m >= 0.5).astype(np.uint8) * 255
            elif m.dtype == np.bool_:
                m = m.astype(np.uint8) * 255
            elif m.dtype == np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            else:
                m = (m != 0).astype(np.uint8) * 255

            # 尺寸对齐
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

            # 只要该点在掩码内才候选
            if m[py, px] > 0:
                area = int(m.sum())
                if area > best_area:
                    best_area = area
                    obj_mask = m

        if best_area < 0:
            print(f"⚠️ SAM 未找到包含点({px},{py})的目标，输出将只受ROI与深度限制影响。")

    else:
        print("⚠️ SAM 未返回有效掩码，输出将只受ROI与深度限制影响。")

    # --- 4) 三者求交：(ROI ∩ SAM物体掩码 ∩ 深度有效) ---
    final = cv2.bitwise_and(roi, obj_mask)
    final = cv2.bitwise_and(final, mask_depth)   # 仍为 0/255 单通道

    # --- 5) 保存 ---
    cv2.imwrite(save_path, final, [cv2.IMWRITE_PNG_BILEVEL, 1])
    print(f"✅ 最终掩码已保存：{save_path}（形式为 ROI ∩ SAM物体掩码 ∩ 深度）")
    print(f"[DBG] radius_px={radius_px}, annulus={annulus}")
    print(f"[DBG] ROI sum={int(roi.sum())}")             # >0 才说明圆/环画出来了
    print(f"[DBG] obj_mask sum={int(obj_mask.sum())}")   # >0 才说明SAM选到了包含点的目标
    print(f"[DBG] depth mask sum={int(mask_depth.sum())}")  # 若用深度，应>0
    return final

def get_circular_area_mask(
    image_bgr,
    point_xy,
    radius_px=30,
    depth=None,
    depth_max=2.0
):
    """
    不使用 SAM 模型，直接返回指定半径圆内的所有区域掩码。
    """
    H, W = image_bgr.shape[:2]
    px, py = int(round(point_xy[0])), int(round(point_xy[1]))
    
    # 1. 生成圆形 ROI (这就是你要的“全部范围”)
    # 使用你原本代码里的辅助函数 _make_circular_mask
    mask_roi = _make_circular_mask((H, W), (px, py), radius_px) 

    # 2. (可选) 深度过滤 - 防止抓取到过远的地板
    mask_depth = np.ones((H, W), dtype=np.uint8) * 255
    if depth is not None:
        d = np.asarray(depth)
        if d.shape[:2] != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_depth = ((d < float(depth_max))).astype(np.uint8) * 255
    
    # 3. 最终掩码 = 圆形区域 & 深度限制 (不再 & SAM物体掩码)
    final_mask = cv2.bitwise_and(mask_roi, mask_depth)
    
    return final_mask
if __name__ == '__main__':
    seg_mask = segment_and_crop_mask('color_img_path.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
