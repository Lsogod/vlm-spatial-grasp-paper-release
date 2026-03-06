# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import io
import re
import json
import math
import time
import base64
import argparse
import numpy as np
from typing import List, Dict, Literal, Tuple, Optional, Union
from collections import OrderedDict, defaultdict

import requests
from dotenv import load_dotenv  
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI, AzureOpenAI

load_dotenv()

def _mask(s, keep=6):
    if not s: return "<EMPTY>"
    s = s.strip()
    return s[:keep] + "..." + s[-4:] + f" (len={len(s)})"

def _get_client():
    load_dotenv()
    base_url = (os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1") or "").rstrip("/")
    api_key  = (os.getenv("MODELSCOPE_API_KEY") or "").strip()
    model_id = os.getenv("MODELSCOPE_MODEL_ID", "qwen3-vl-plus-2025-12-19")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_id
def _get_azure_client():
    """初始化 Azure OpenAI 客户端"""
    load_dotenv()

    api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not api_key or not endpoint:
        raise RuntimeError("请先设置 AZURE_OPENAI_API_KEY 与 AZURE_OPENAI_ENDPOINT")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return client, deployment
# 基础：读图 & 可视化
def load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith(("http://", "https://")):
        resp = requests.get(image_path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 新增：支持 data URL（data:image/...;base64,XXXX）
    if image_path_or_url.startswith("data:image/"):
        try:
            header, b64 = image_path_or_url.split(",", 1)
        except ValueError:
            raise ValueError("data URL 格式错误：缺少逗号分隔头部与数据")
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")

    # 默认：按本地文件路径处理
    return Image.open(image_path_or_url).convert("RGB")


def draw_bbox_on_image_pil(
    image_path_or_url: str,
    bbox_xyxy: List[int],
    output_path: Optional[str] = None,
    color: str = "lime",
    width: int = 3,
) -> str:
    """
    在图像上绘制边界框（像素坐标）并保存（纯 PIL 实现）
    bbox_xyxy: [x1, y1, x2, y2] 像素坐标
    """
    img = load_image(image_path_or_url)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3]))],
                   outline=color, width=width)
    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"annotated_{ts}.jpg"
    img.save(output_path)
    return output_path


# 文本清理/解析工具
def strip_code_fences(text: str) -> str:
    text = re.sub(r"```json\s*", "", text, flags=re.I)
    text = re.sub(r"```", "", text)
    return text.strip()


def extract_first_json_array(text: str) -> Optional[str]:
    text = strip_code_fences(text)
    start = text.find("[")
    if start == -1:
        return None
    end = text.rfind("]")
    if end == -1 or end < start:
        return None
    return text[start:end + 1]

# 坐标与校验
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def validate_and_fix_bbox_item(item: dict) -> Optional[dict]:
    """
    规范化单个 bbox 项：
      - 必须含 'bbox_2d' 和 'label'
      - 坐标四个数都在 [0,1000]，若超界则裁剪
      - 若 x1>x2 或 y1>y2 则交换
    """
    if not isinstance(item, dict):
        return None
    if "bbox_2d" not in item or "label" not in item:
        return None
    bb = item["bbox_2d"]
    if not (isinstance(bb, list) and len(bb) == 4):
        return None
    x1, y1, x2, y2 = bb
    try:
        x1 = clamp(float(x1), 0, 1000)
        y1 = clamp(float(y1), 0, 1000)
        x2 = clamp(float(x2), 0, 1000)
        y2 = clamp(float(y2), 0, 1000)
    except Exception:
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    item["bbox_2d"] = [x1, y1, x2, y2]
    item["label"] = str(item.get("label", ""))
    return item


def validate_and_fix_point_item(item: dict) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    if "point_2d" not in item or "label" not in item:
        return None
    pt = item["point_2d"]
    if not (isinstance(pt, list) and len(pt) == 2):
        return None
    try:
        x = clamp(float(pt[0]), 0, 1000)
        y = clamp(float(pt[1]), 0, 1000)
    except Exception:
        return None
    item["point_2d"] = [x, y]
    item["label"] = str(item.get("label", ""))
    return item


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def nms_sorted(items: List[dict], iou_thresh=0.7) -> List[dict]:
    def area(it):
        x1, y1, x2, y2 = it["bbox_2d"]
        return (x2 - x1) * (y2 - y1)
    items = sorted(items, key=area)  # 小框优先
    keep = []
    for it in items:
        bx = it["bbox_2d"]
        if any(iou_xyxy(bx, kt["bbox_2d"]) > iou_thresh for kt in keep):
            continue
        keep.append(it)
    return keep


# 相对/像素坐标换算（可选）
def rel_to_abs_point(pt_rel, W, H):
    x = int(round(pt_rel[0] / 1000.0 * W))
    y = int(round(pt_rel[1] / 1000.0 * H))
    x = clamp(x, 0, W - 1)
    y = clamp(y, 0, H - 1)
    return int(x), int(y)


def rel_to_abs_xyxy(bb_rel, W, H):
    x1 = int(round(bb_rel[0] / 1000.0 * W))
    y1 = int(round(bb_rel[1] / 1000.0 * H))
    x2 = int(round(bb_rel[2] / 1000.0 * W))
    y2 = int(round(bb_rel[3] / 1000.0 * H))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = clamp(x1, 0, W - 1)
    x2 = clamp(x2, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    y2 = clamp(y2, 0, H - 1)
    return int(x1), int(y1), int(x2), int(y2)


def draw_bboxes(img: Image.Image, items: List[dict]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except Exception:
        font = ImageFont.load_default()
    W, H = img.size
    palette = ["red","green","blue","yellow","orange","pink","purple","cyan","magenta","lime","navy","teal"]
    print("=== 绘制边界框坐标（像素） ===")
    for i, it in enumerate(items):
        # 将相对坐标 [0,1000] 转换为像素坐标
        rel_x1, rel_y1, rel_x2, rel_y2 = it["bbox_2d"]
        x1 = int(rel_x1 / 1000 * W)
        y1 = int(rel_y1 / 1000 * H)
        x2 = int(rel_x2 / 1000 * W)
        y2 = int(rel_y2 / 1000 * H)

        color = palette[i % len(palette)]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        label = str(it.get("label", ""))
        extra = []
        if "type" in it: extra.append(str(it["type"]))
        if "color" in it: extra.append(str(it["color"]))
        if extra: label += " (" + ", ".join(extra) + ")"
        if label:
            draw.text((x1 + 6, y1 + 4), label, fill=color, font=font)

        print(f"→ {label or '(无标签)'}: 左上 ({x1}, {y1}), 右下 ({x2}, {y2})")

    print("================================")
    return img

def draw_points(img: Image.Image, items: List[dict]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except Exception:
        font = ImageFont.load_default()
    W, H = img.size
    palette = ["red","green","blue","yellow","orange","pink","purple","cyan","magenta","lime","navy","teal"]
    print("=== 绘制点坐标（像素） ===")
    for i, it in enumerate(items):
        # 将相对坐标转换为像素坐标
        rel_x, rel_y = it["point_2d"]
        x = int(rel_x / 1000 * W)
        y = int(rel_y / 1000 * H)

        color = palette[i % len(palette)]
        r = max(2, int(min(W, H) * 0.004))
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)

        label = str(it.get("label", ""))
        if "role" in it: label += f" ({it['role']})"
        if "shirt_color" in it: label += f" [{it['shirt_color']}]"
        if label:
            draw.text((x + 2 * r, y + 2 * r), label, fill=color, font=font)

        print(f"→ {label or '(无标签)'}: ({x}, {y})")

    print("========================")
    return img



# 统一模型调用（多模态）
def call_model_api(image_path_or_url: str, prompt: str) -> str:
    """
    使用 OpenAI 兼容接口调用多模态模型（默认 Qwen3-VL）
    返回模型输出的纯文本字符串（模型需按指令输出 JSON）
    """
    client, model_id = _get_client()

    # 构造 image_url（支持本地/远程）
    # if os.path.exists(image_path_or_url):
    #     with open(image_path_or_url, "rb") as f:
    #         base64_image = base64.b64encode(f.read()).decode("utf-8")
    #     image_url = f"data:image/jpeg;base64,{base64_image}"
    # elif image_path_or_url.startswith(("http://", "https://")):
    #     image_url = image_path_or_url
    # else:
    #     raise ValueError("图片路径或URL无效")

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_path_or_url}
            },
            {"type": "text", "text": prompt},
        ],
    }]

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.1,
    )
    return completion.choices[0].message.content



# 端到端 Demo：BBox / Point
def parse_boxes_from_text(text: str) -> List[dict]:
    candidate = extract_first_json_array(text)
    if candidate is None:
        raise ValueError("未找到 JSON 数组")
    data = json.loads(candidate)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是数组")
    return data


def parse_points_from_text(text: str) -> List[dict]:
    candidate = extract_first_json_array(text)
    if candidate is None:
        raise ValueError("未找到 JSON 数组")
    data = json.loads(candidate)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是数组")
    return data


def run_bbox_demo(image_path_or_url: str, categories: List[str],
                  with_attrs=False, max_instances=50, iou_dedupe=0.7):
    if with_attrs:
        prompt = (
            'Locate every instance in categories: "' + ", ".join(categories) + '". '
            'Return a JSON array; each item MUST be: '
            '{"bbox_2d":[x1,y1,x2,y2], "label":"<category>", "type":"<optional>", "color":"<optional>"}; '
            'Coordinates are RELATIVE integers in [0,1000]. '
            f'Max {max_instances} items. No extra text.'
        )
    else:
        prompt = (
            'Locate every instance in categories: "' + ", ".join(categories) + '". '
            'Return a JSON array of objects: {"bbox_2d":[x1,y1,x2,y2], "label":"<category>"}; '
            'Coordinates are RELATIVE integers in [0,1000]. '
            f'Max {max_instances} items. No extra text.'
        )

    raw_text = call_model_api(image_path_or_url, prompt)
    items = parse_boxes_from_text(raw_text)

    fixed = []
    for it in items:
        ok = validate_and_fix_bbox_item(it)
        if ok is not None:
            fixed.append(ok)
    fixed = nms_sorted(fixed, iou_thresh=iou_dedupe)

    img = load_image(image_path_or_url)
    img.thumbnail((960, 960))
    vis = draw_bboxes(img, fixed)
    return fixed, vis

def run_point_demo(image_path_or_url: str, who="person",
                   with_attrs=False, max_points=80):
    if with_attrs:
        prompt = (
            f'Locate every {who} with POINTS. '
            'Return a JSON array; each item MUST be: '
            '{"point_2d":[x,y], "label":"person", "role":"player/referee/unknown", "shirt_color":"..."}; '
            'Coordinates are RELATIVE integers in [0,1000]. '
            f'Max {max_points} items. No extra text.'
        )
    else:
        prompt = (
            f'Locate every {who} with POINTS. '
            'Return a JSON array of objects: {"point_2d":[x,y], "label":"person"}; '
            'Coordinates are RELATIVE integers in [0,1000]. '
            f'Max {max_points} items. No extra text.'
        )

    raw_text = call_model_api(image_path_or_url, prompt)
    items = parse_points_from_text(raw_text)

    fixed = []
    for it in items:
        ok = validate_and_fix_point_item(it)
        if ok is not None:
            fixed.append(ok)

    img = load_image(image_path_or_url)
    img.thumbnail((960, 960))
    vis = draw_points(img, fixed)
    return fixed, vis

# 规划：生成机器人首个动作序列
def generate_first_robot_action(instruction: str, image_url: str) -> str:
    """
    调用 Azure OpenAI（或兼容端点），根据图像和文本指令生成机器人首个动作序列（JSON 串）。
    提示词强化了“相对另一个物体进行放置”的规划能力，并确保：
    - 抓取相关动作（move_to_grasp_pose/close_gripper）的 component 指向被搬运物体；
    - 相对放置的移动动作（move_to_point）的 component 指向参照物，以便写回参照物像素到 args.pixel。
    """
    client, deployment = _get_azure_client()

    system_prompt = (
        "你是一个视觉-操作规划助手。根据用户指令与图像内容，确定需要操作的“具体物体与部位”，"
        "并把整个抓取与放置流程用严格 JSON 表示（不得输出 JSON 之外的任何字符）。\n"
        "\n"
        "【动作集合（只能用以下动词）】\n"
        "1) \"move_to_grasp_pose\": 仅用于抓取时的接近与对准（预抓取→接近→到抓取位姿），且必须紧接着出现一次且仅一次 \"close_gripper\"。\n"
        "2) \"close_gripper\": 闭合夹爪（只有在执行抓取时使用，且前一步必须是 move_to_grasp_pose）。\n"
        "3) \"move_to_point\": 非抓取移动（如搬运、对位、路过中间点），不得用于抓取接近。\n"
        "4) \"open_gripper\": 打开夹爪（通常在放置或释放时使用）。\n"
        "5) \"back_to_zero\": 回到初始位姿，结束任务。\n"
        "\n"
        "【组件标注（非常重要）】\n"
        "- 每条指令的 component 必须为 [\"<object>\", \"<part>\"]，如 [\"banana\",\"middle\"]。\n"
        "- 抓取相关动作（move_to_grasp_pose / close_gripper）：component 指向“被搬运的物体及部位”。\n"
        "- 相对放置的移动动作（move_to_point）：若要将被搬运物体放到“另一个物体的上方/旁边/附近…”，"
        "  component 必须指向该“参照物及部位”（例如 [\"red cup\",\"rim\"]）。这样坐标会来自参照物的位置。\n"
        "- 若用户提到颜色/材质等特征，需体现在 object 中，如 [\"red cup\",\"rim\"]。\n"
        "\n"
        "【相对放置规则（两物体操作）】\n"
        "- 当指令包含“把A放到B的上方/周围/旁边/前后左右/附近”等关系时：\n"
        "  1) 先抓取 A（A 的 middle/center 等部位）。\n"
        "  2) 再执行一条或多条 move_to_point 到参照物 B 的目标部位（如 rim/center/edge），以实现关系放置。\n"
        "  3) 这些 move_to_point 的 component= [\"B\",\"<part>\"]，并可在 args 中给出：\n"
        "     - \"relation\": \"above\" | \"left_of\" | \"right_of\" | \"front_of\" | \"behind\" | \"near\" | \"on_top\"\n"
        "     - \"offset_px\": [dx, dy]  （相对坐标 0~1000 的像素平面微调）\n"
        "     - \"z_offset_m\": 0.03     （垂直方向微调，米）\n"
        "- 如未明确参照部位，默认使用能代表空间关系的合理部位：容器用 \"rim\"，规则物体用 \"center\"，桌面用 \"surface/center\" 等。\n"
        "- 若同类参照有多个候选，优先选择更靠近图像中心、无遮挡、尺度合适者，并在 description 简述理由。\n"
        "\n"
        "【时序/一致性约束（必须满足）】\n"
        "- 如出现 \"close_gripper\"，其前一步必须且只能是一次 \"move_to_grasp_pose\"；\n"
        "- 不允许用 \"move_to_point\" 替代抓取接近；\n"
        "- 若出现 \"open_gripper\"，通常应在一次或多次 \"move_to_point\" 之后；\n"
        "- \"back_to_zero\" 必须在流程末尾且只出现一次；\n"
        "- 整个流程可以涉及 2 个或以上物体；对“把A移动到B上方/周围”等需求，必须体现出对 A 的抓取与对 B 的相对放置。\n"
        "\n"
        "【输出格式（严格）】\n"
        "仅输出一个 JSON 对象：{\"action_components\": [...]}，不得包含 JSON 以外的任何文本或注释。\n"
        "数组内每项结构：\n"
        "{\n"
        "  \"component\": [\"<object>\", \"<part>\"],\n"
        "  \"action\": \"move_to_grasp_pose | close_gripper | move_to_point | open_gripper | back_to_zero\",\n"
        "  \"description\": \"用中文简述该步目的与原因（如：为什么选择该参照物、该关系）\"\n"
        "}\n"
        "\n"
        "【严格示例（仅示意；真实坐标由你根据图像推断）】\n"
        "{\n"
        "  \"action_components\": [\n"
        "    {\"component\": [\"yellow duck\",\"middle\"], \"action\": \"move_to_grasp_pose\", \"description\": \"对准小黄鸭中部以便稳定抓取\"},\n"
        "    {\"component\": [\"yellow duck\",\"middle\"], \"action\": \"close_gripper\", \"description\": \"夹紧小黄鸭\"},\n"
        "    {\"component\": [\"red cup\",\"rim\"],        \"action\": \"move_to_point\",    \"description\": \"移动到红杯杯沿正上方\"},\n" 
        "    {\"component\": [\"yellow duck\",\"middle\"], \"action\": \"open_gripper\",     \"description\": \"释放小黄鸭\"},\n"
        "    {\"component\": [\"arm\",\"base\"],           \"action\": \"back_to_zero\",     \"description\": \"回零结束\"}\n"
        "  ]\n"
        "}\n"
        "\n"
        "不要出现move_to_point动作重复对一个物体组件操作\n"
        "——请直接输出最终 JSON，不要解释。"
    )
 
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": instruction},
                ],
            },
        ],
        temperature=0.1,
        stream=False
    )
    return response.choices[0].message.content

# 目标类型推断 & 提取
ReturnType = Literal["point", "bbox", "auto"]

def infer_return_type(obj: str, part: str) -> str:
    """
    简单启发式：
      - 点状部位 → point：center, middle, tip, mouth, spout, button, centroid
      - 区域/边界部位 → bbox：handle, rim, edge, corner, surface, region, area
    """
    part_l = str(part).lower()
    point_like = {"center", "middle", "tip", "mouth", "spout", "button", "centroid"}
    bbox_like  = {"handle", "rim", "edge", "corner", "surface", "region", "area"}
    if part_l in point_like:
        return "point"
    if part_l in bbox_like:
        # 若希望把 handle 也作为点返回，改为 "point"
        return "bbox"
    return "point"


def extract_targets_from_actions(action_json: str) -> List[Tuple[str, str, str]]:
    data = json.loads(action_json)
    acts = data.get("action_components", [])
    uniq = OrderedDict()
    for a in acts:
        comp = a.get("component")
        if not (isinstance(comp, list) and len(comp) == 2):
            continue
        obj, part = comp[0], comp[1]
        # 过滤机械臂自身
        if str(obj).lower() in {"arm", "robot", "manipulator"}:
            continue
        key = (str(obj), str(part))
        if key not in uniq:
            uniq[key] = infer_return_type(obj, part)
        else:
            # 若后续遇到不同推断，偏向 bbox（更通用）
            if uniq[key] != "bbox":
                if infer_return_type(obj, part) == "bbox":
                    uniq[key] = "bbox"
    return [(obj, part, rtype) for (obj, part), rtype in uniq.items()]

def locate_multiple_parts(
    image_path_or_url: str,
    targets: List[Tuple[str, str, ReturnType]],
    max_items_per_target: int = 1,
    disambiguation: Optional[str] = None,
    context: Optional[str] = None,
    bbox_size_hint: int = 60
) -> Tuple[List[Dict], Image.Image]:

    generic_defs = (
        "- 'center' of round/elliptical/open container (e.g., bowl/cup): the geometric center of the visible opening interior.\n"
        "- 'middle' of elongated/curved objects (e.g., banana): the midpoint along the main axis from one end to the other.\n"
        "- If multiple instances exist, return up to the specified max per target.\n"
    )
    if disambiguation:
        generic_defs += f"- Disambiguation hint: {disambiguation}\n"
    if context:
        generic_defs += f"- Scene context: {context}\n"

    lines = []
    for obj, part, rtype in targets:
        if rtype == "point":
            lines.append(f'- Object "{obj}", Part "{part}", Return "point" (coordinates in [0,1000]).')
        elif rtype == "bbox":
            lines.append(f'- Object "{obj}", Part "{part}", Return "bbox" (tight small bbox, ~{bbox_size_hint}/1000).')
        else:
            lines.append(f'- Object "{obj}", Part "{part}", Return "auto" (choose point for spots; bbox for areas).')
    target_block = "\n".join(lines)

    schema = (
        "Return a single JSON array. Each item MUST be one of the following forms:\n"
        '  {"object":"<object_name>", "part":"<part_name>", "return_type":"point", "point_2d":[x,y]}\n'
        '  {"object":"<object_name>", "part":"<part_name>", "return_type":"bbox",  "bbox_2d":[x1,y1,x2,y2]}\n'
        "Constraints:\n"
        "- Coordinates are RELATIVE integers in [0,1000].\n"
        f"- For each (object, part), return AT MOST {max_items_per_target} items.\n"
        "- Do NOT include any explanations outside the JSON array."
    )

    prompt = (
        "Locate the following object parts in the image:\n"
        f"{target_block}\n\n"
        "Guidelines:\n"
        f"{generic_defs}\n"
        f"{schema}"
    )

    raw_text = call_model_api(image_path_or_url, prompt)

    candidate = extract_first_json_array(raw_text)
    if candidate is None:
        print("Model raw output:\n", raw_text)
        raise ValueError("未找到 JSON 数组")

    data = json.loads(candidate)
    if not isinstance(data, list):
        print("Model raw output:\n", raw_text)
        raise ValueError("JSON 顶层必须是数组")

    normalized: List[Dict] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        obj = str(it.get("object", "")).strip()
        part = str(it.get("part", "")).strip()
        rtype = str(it.get("return_type", "")).strip().lower()

        if not obj or not part or rtype not in {"point", "bbox"}:
            if rtype in {"auto", ""}:
                if "point_2d" in it:
                    rtype = "point"
                elif "bbox_2d" in it:
                    rtype = "bbox"
                else:
                    continue
            else:
                continue

        if rtype == "point" and "point_2d" in it:
            ok = validate_and_fix_point_item({"point_2d": it["point_2d"], "label": f"{part} of {obj}"})
            if ok:
                ok["object"] = obj
                ok["part"] = part
                ok["return_type"] = "point"
                normalized.append(ok)

        elif rtype == "bbox" and "bbox_2d" in it:
            ok = validate_and_fix_bbox_item({"bbox_2d": it["bbox_2d"], "label": f"{part} of {obj}"})
            if ok:
                ok["object"] = obj
                ok["part"] = part
                ok["return_type"] = "bbox"
                normalized.append(ok)

    # 分离 point / bbox，并对 bbox 分组 NMS + 截断
    bboxes = [x for x in normalized if x["return_type"] == "bbox"]
    points = [x for x in normalized if x["return_type"] == "point"]

    groups = defaultdict(list)
    for bb in bboxes:
        groups[(bb["object"], bb["part"])].append(bb)

    deduped_bboxes: List[Dict] = []
    for _, group in groups.items():
        cleaned = nms_sorted(group, iou_thresh=0.7)
        cleaned = cleaned[:max_items_per_target]
        deduped_bboxes.extend(cleaned)

    point_groups = defaultdict(list)
    for pt in points:
        point_groups[(pt["object"], pt["part"])].append(pt)

    deduped_points: List[Dict] = []
    for _, group in point_groups.items():
        deduped_points.extend(group[:max_items_per_target])

    final_items = deduped_points + deduped_bboxes

    # 可视化
    img = load_image(image_path_or_url)
    img.thumbnail((960, 960))
    if deduped_bboxes:
        img = draw_bboxes(img, deduped_bboxes)
    if deduped_points:
        img = draw_points(img, deduped_points)

    return final_items, img

def plan_and_ground_robot_actions(
    instruction: str,
    image_path: str,
    *,
    max_items_per_target: int = 1,
    disambiguation: str = "prefer the most central instance if multiple",
    context: Optional[str] = None,
    bbox_size_hint: int = 70,
    save_vis_prefix: Optional[str] = None,
) -> Dict:
    """
    1) 规划动作 JSON
    2) 抽取 (object, part, return_type)
    3) 多部位定位
    4) 将坐标写回动作 JSON 的 args.pixel
    """
    # A. 规划
    image_path_or_url = to_data_url(image_path)
    orig_img = load_image_any(image_path)
    W, H = orig_img.size
    raw_plan_str = generate_first_robot_action(instruction=instruction, image_url=image_path_or_url)
    try:
        plan = json.loads(raw_plan_str)
        # print("[INFO] 规划动作 JSON 解析成功")
        # print(raw_plan_str)
    except Exception as e:
        raise RuntimeError(f"动作规划 JSON 解析失败：{e}\n原始文本：{raw_plan_str[:500]}...")

    # B. 目标抽取
    targets = extract_targets_from_actions(raw_plan_str)
    if not targets:
        return {
            "action_json_str": json.dumps(plan, ensure_ascii=False, indent=2),
            "action_json": plan,
            "detections": [],
            "vis_path": None,
            "raw_plan": raw_plan_str,
        }

    # C. 批量定位
    detections, vis_img = locate_multiple_parts(
        image_path_or_url=image_path_or_url,
        targets=targets,
        max_items_per_target=max_items_per_target,
        disambiguation=disambiguation,
        context=context,
        bbox_size_hint=bbox_size_hint,
    )

    vis_path = None
    if save_vis_prefix:
        os.makedirs(os.path.dirname(save_vis_prefix), exist_ok=True)
        vis_path = f"{save_vis_prefix}_detections.png"
        vis_img.save(vis_path)

    # D. 将检测结果组织为候选池（相对坐标）
    def bbox_center(b):
        x1, y1, x2, y2 = map(float, b)
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def as_key(obj, part):
        return (str(obj).strip().lower(), str(part).strip().lower())

    candidates: Dict[Tuple[str, str], List[Dict]] = {}
    for det in detections:
        obj = det.get("object", "")
        part = det.get("part", "")
        rtype = str(det.get("return_type", "")).lower()
        if not obj or not part:
            continue
        px = None
        if rtype == "point" and "point_2d" in det:
            px = [float(det["point_2d"][0]), float(det["point_2d"][1])]
        elif rtype == "bbox" and "bbox_2d" in det:
            px = bbox_center(det["bbox_2d"])
        if px is None:
            continue
        world = det.get("world")
        candidates.setdefault(as_key(obj, part), []).append({"pixel": px, "world": world})

    # E. 写回动作中的 args.pixel（相对坐标 0-1000）
    def rel_dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def rel_to_abs_xy(px_rel):  # 相对 → 像素
        x_px = int(round(px_rel[0] / 1000.0 * W))
        y_px = int(round(px_rel[1] / 1000.0 * H))
        return [clamp(x_px, 0, W - 1), clamp(y_px, 0, H - 1)]

    def to_rel_xy(px_or_rel):   # 距离比较用：把已有值统一转成“相对坐标”
        x, y = float(px_or_rel[0]), float(px_or_rel[1])
        # 经验判断：若坐标分量大于1000，视为像素坐标，转为相对坐标
        if max(x, y) > 1000:
            return [x * 1000.0 / W, y * 1000.0 / H]
        return [x, y]

    REF_CENTER_REL = [500.0, 500.0]

    for ac in plan.get("action_components", []):
        comp = ac.get("component")
        if not (isinstance(comp, list) and len(comp) == 2):
            continue
        key = as_key(comp[0], comp[1])
        pool = candidates.get(key, [])
        if not pool:
            continue

        ac.setdefault("args", {})
        existing = ac["args"].get("pixel")

        if isinstance(existing, (list, tuple)) and len(existing) == 2:
            base_rel = to_rel_xy(existing)
            best = min(pool, key=lambda c: rel_dist(c["pixel"], base_rel))
        else:
            best = min(pool, key=lambda c: rel_dist(c["pixel"], REF_CENTER_REL))

        # 将“相对坐标”转换为“像素坐标”后写入
        px_abs = rel_to_abs_xy(best["pixel"])
        ac["args"]["pixel"] = px_abs  # 现在是像素单位

        if best.get("world") is not None:
            ac["args"]["world"] = best["world"]

    final_str = json.dumps(plan, ensure_ascii=False, indent=2)
    return {
        "action_json_str": final_str,
        "action_json": plan,
        "detections": detections,
        "vis_path": vis_path,
        "raw_plan": raw_plan_str,
    }

def to_data_url(image: Union[str, np.ndarray, Image.Image], *, fmt: str = "JPEG") -> str:
    # 已是字符串
    if isinstance(image, str):
        if image.startswith("data:image/"):
            return image  # 已是 data URL
        if image.startswith(("http://", "https://")):
            return image  # 远程 URL：保留URL给模型（多数端点支持 http(s)）
        if os.path.exists(image):
            # 本地文件 → data URL
            with open(image, "rb") as f:
                b = f.read()
            b64 = base64.b64encode(b).decode("utf-8")
            mime = _guess_mime_from_ext(os.path.splitext(image)[-1])
            return f"data:{mime};base64,{b64}"
        # 普通字符串但不是路径/URL/data:，作为错误处理
        raise ValueError(f"无法识别的图像字符串输入：{image[:64]}...")

    # 是 PIL.Image
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        # PNG时带alpha，其他一律转RGB
        im = image if (fmt.upper() == "PNG" and image.mode in ("RGBA", "LA")) else image.convert("RGB")
        im.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"

    # 是 numpy.ndarray
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:  # 灰度
            im = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # 猜测 BGR → RGB（OpenCV出来通常是BGR）
            # 检测方式：直接视为BGR再翻转到RGB，效果更稳
            im = Image.fromarray(arr[:, :, ::-1].astype(np.uint8), mode="RGB")
        else:
            raise ValueError(f"不支持的 ndarray 形状：{arr.shape}")
        buf = io.BytesIO()
        im.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"

    raise TypeError(f"不支持的图像输入类型：{type(image)}")
def _guess_mime_from_ext(ext: str) -> str:
    ext = ext.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(ext, "image/jpeg")



# 工具2：把任意图像输入统一为 PIL.Image（给可视化/绘制）
# 支持 data URL、http(s) URL、文件路径、ndarray、PIL.Image
def load_image_any(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr[:, :, ::-1].astype(np.uint8), mode="RGB")  # BGR→RGB
        raise ValueError(f"不支持的 ndarray 形状：{arr.shape}")

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            import requests
            resp = requests.get(image, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        if image.startswith("data:image/"):
            try:
                header, b64 = image.split(",", 1)
            except ValueError:
                raise ValueError("data URL 格式错误：缺少逗号分隔头部与数据")
            data = base64.b64decode(b64)
            return Image.open(io.BytesIO(data)).convert("RGB")
        if os.path.exists(image):
            return Image.open(image).convert("RGB")

    raise ValueError(f"无法从该输入读取图像：{type(image)}")

# 命令行演示
def main():
    # ap = argparse.ArgumentParser(description="机器人视觉-操作一体化示例")
    # ap.add_argument("--image", required=True, help="图片路径或URL（本地/HTTP均可）")
    # ap.add_argument("--instruction", required=True, help="中文任务指令")
    # ap.add_argument("--out_prefix", default=None, help="可视化保存前缀，例如 logs/run1（会保存 *_detections.png）")
    # args = ap.parse_args()
    # image_path = "/root/VLM_Grasp_Interactive/logs/input_images/input_to_model_20251106_205422.jpg"
     # args.image
    res = plan_and_ground_robot_actions(
        instruction="抓取香蕉和锤子小黄鸭",
        image_path=color_img,
        save_vis_prefix="logs/run1",
        context="tabletop scene"
    )
    print("\n=== 最终动作（已写回坐标） ===")
    print(res["action_json_str"])
    if res["vis_path"]:
        print(f"\n可视化已保存：{res['vis_path']}")


if __name__ == "__main__":
    main()
