# vision_agent_v2.py
# -*- coding: utf-8 -*-
"""
vision_agent_v2.py

目的：
- 用“前端编排逻辑”的同款三阶段提示词（Stage1/Stage2/Stage3）完成：
  用户指令 -> 动作规划 -> 编号映射 -> 像素落点写回 args.pixel
- 与旧版 plan_and_ground_robot_actions() 对齐：
  返回 res["action_json_str"], res["detections"], res["vis_path"]

新增：
- 每一步增加 JSONL 日志（包含输入/输出、关键中间变量）
- 命令行参数：--log logs/run1.jsonl (可选)
  默认不写 base64/data_url，只写长度与 hash 摘要，避免日志爆炸
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import base64
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None



# ============================================================
# ⚙️ 配置：后端与 Azure OpenAI（你可以用环境变量覆盖）
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5-chat")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_KEY = os.getenv("AZURE_KEY", "")  # 强烈建议放环境变量里


# ============================================================
# 🛑 三段提示词（与你前端版本一致，可按需微调）
# ============================================================

PROMPT_STAGE_1 = """你需要根据用户上传的图片内容和用户的自然语言指令，分析场景中需要进行操作的物体，识别这些物体上的数字编号，并推理出操作约束。

要求：
1) 先从用户指令中推断出明确的 user_intent（例如“我有点口渴”→“想喝水/倒水进杯子”）。
2) 在图片中定位与该意图相关的物体；若同一物体由多个编号标注其不同组件（如壶身、壶盖、旋钮），   则必须将这些编号全部收集并以组件的形式输出。
3) 如果需要进行操作的物体上有多个编号，请全部选择；若一个动作需要多个物体配合（如水壶与杯子），请都输出。
4) 推理并给出操作约束，包含：合理的操作顺序、是否需要先打开/取下组件、是否需要同时使用、   搬动/倒置/放置等动作说明；如存在安全/防溢出等注意事项也可简要指出。
5) 仅输出 JSON，且必须符合下述结构与字段命名；不要输出除 JSON 外的任何文字。

输出 JSON 结构：
{
  "user_intent": "<用户意图>",
  "objects": [
    {
      "name": "<物体名称>",
      "components": [
        {"id": <编号>, "part": "<组件名称>"}
        // 如有多个组件，继续追加
      ]
    }
    // 如有多个物体，继续追加
  ],
  "constraints": "<操作约束与顺序说明>"
}

示例（仅示意，需基于真实图片识别填充）：
{
  "user_intent": "将水壶中的水倒入杯子中",
  "objects": [
    {
      "name": "水壶",
      "components": [
        {"id": 3,  "part": "壶身"},
        {"id": 11, "part": "壶盖"},
        {"id": 6,  "part": "壶盖旋钮"}
      ]
    },
    {
      "name": "杯子",
      "components": [
        {"id": 1, "part": "杯体"}
      ]
    }
  ],
  "constraints": "先取下编号(11,6)以打开编号3的水壶，再将壶身中的水缓慢倒入编号1的杯子中，控制倾斜角度避免溢出。"
}
"""

PROMPT_STAGE_2 = """你是一个视觉-操作规划助手。你的任务是根据用户指令与图像内容，规划机械臂动作，并输出严格 JSON（不得包含 JSON 以外的任何文本）。每条动作的 component 必须是二元数组 ["<object>", "<part>"]，例如 ["banana","middle"]。

========================
【最重要：抓取部位有效性（硬规则，必须遵守）】
========================
A) 任何“抓取相关动作”（move_to_grasp_pose / close_gripper）选择的 <part> 必须位于该物体的“实体材质区域（solid/material surface）”上。
B) 若该物体存在空洞/镂空/孔洞/空腔（例如胶带卷内孔、杯口空腔、环形物体中心孔、碗盆内部空腔），则：
   - 禁止把空洞/内孔/空腔区域当作可抓取部位；
   - 禁止输出 part = "center"/"inside"/"hole"/"inner" 这类会指向空洞/中心空腔的部位；
   - 必须改选“实体材质表面”的部位：例如 outer_ring / outer_edge / rim / sidewall / handle / body / middle(实体部分) 等。
C) 对环形/中空物体的硬规则（专门避免内外圈误判）：
   1) 胶带卷/圆环/杯口/盘子边缘/碗盆：优先选择 outer_ring（外圈实体）或 sidewall（侧壁实体）或 rim（杯沿/盆沿实体），绝不选择 center。
   2) 如果不确定某个区域是否是实体材质（存在歧义），宁可选择 outer_edge/sidewall 这类更保守的实体部位，也不要选 center。

========================
【输出结构（严格 JSON）】
========================
只输出一个 JSON 对象：{"action_components": [...]}，不得出现任何解释文字。

每个动作项结构为：
{
  "component": ["<object>", "<part>"],
  "action": "<从下列动作中选择>",
  "description": "用中文简述该步目的与选择原因（可写避障/稳定性理由，但不新增动词）"
}

动作集合（只能使用这些动词）：
- "move_to_grasp_pose": 仅用于抓取时的接近与对准（预抓取→接近→到抓取位姿），且必须紧接着出现一次且仅一次 "close_gripper"。
- "close_gripper": 闭合夹爪（只有在执行抓取时使用，且前一步必须是 move_to_grasp_pose）。
- "move_to_point": 非抓取移动（如搬运、对位、移动到放置点上方），不得用于抓取接近。
- "open_gripper": 打开夹爪（通常在放置或释放时使用）。
- "back_to_zero": 回到初始位姿，结束任务。

========================
【选择策略】
========================
1) 若某目标出现多个候选，优先选择：更靠近图像中心、无遮挡、尺度合适、易抓取且更稳定的实体材质区域。
2) 若指令包含颜色/材质/大小等描述，请据此选择对应物体，并在 object 或 part 中体现（例如 "red cup"、"tape outer_ring"）。
3) part 命名建议（不强制，但建议遵循以便后续映射）：
   - 可抓实体中心：middle / body（仅当确认为“实心/实体中心”）
   - 环形/中空：outer_ring / outer_edge / sidewall / rim（优先用这些）
   - 杯子/碗/盆：rim（沿）/ sidewall（侧壁外侧）/ body（外壁）
   - 带把手：handle
   - 细长物体：middle（中段实体）/ end（端部实体）

========================
【严格示例（仅示意格式）】
========================
场景 A：单纯抓取任务 (Pick Only)
{
  "action_components": [
    {"component": ["banana","middle"], "action": "move_to_grasp_pose", "description": "对准香蕉中段实体表面，稳定且不易滑落"},
    {"component": ["banana","middle"], "action": "close_gripper",      "description": "夹紧香蕉"},
    {"component": ["arm","base"],      "action": "back_to_zero",       "description": "抓取成功后回零"}
  ]
}

【关键示例：胶带卷（环形中空，禁止 center）】
{
  "action_components": [
    {"component": ["double-sided tape","outer_ring"], "action": "move_to_grasp_pose", "description": "胶带为环形中空物体，中心为空洞不可抓，选择外圈实体材质区域更稳"},
    {"component": ["double-sided tape","outer_ring"], "action": "close_gripper",      "description": "夹紧外圈实体表面"},
    {"component": ["arm","base"],                     "action": "back_to_zero",       "description": "抓取成功后回零"}
  ]
}

========================
【自检（必须在输出前完成）】
========================
- 对于每一对 move_to_grasp_pose/close_gripper：检查其 part 是否指向空洞/内孔/空腔；
  若是（尤其是环形/中空物体），必须改为 outer_ring / rim / sidewall / outer_edge 等实体表面部位后再输出。

——请直接输出最终 JSON，不要解释，不要加注释，不要加 Markdown。
"""

PROMPT_STAGE_3_TEMPLATE = """你将收到一张带红白数字编号的图片，以及一个目标列表 targets，
例如：targets = [('kettle','knob','point'), ('cup','rim','point'), ('kettle','spout','point')]。
每个三元组含义为：(物体, 部位, 选择类型)，其中选择类型为 'point'，表示需要选出该部位上“一个最相关的编号”。

你的任务：为 targets 中的每个目标在图片里找到对应部位上的编号；若该部位没有贴编号，则按以下就近规则选择最近编号。

【规则】
1) 只能使用图片中真实可见的红白数字编号，禁止编造。
2) 优先在“该部位”上选编号；若该部位上有多个编号，选与该部位几何中心距离最近的编号。
3) 若“该部位”没有编号，退而在“同一物体的其它部位”中，选择与该部位位置最近的编号。
4) 若整件物体都没有编号，则在整张图中选择与该部位所在区域最近的编号（全图最近）。
5) 若无法确定该物体是否存在或严重遮挡，返回 id 为 null。
6) 仅输出编号，不要改变输入目标的顺序，不要输出多余文字。

【输出格式（严格 JSON）】
{
  "results": [
    {"object": "<物体>", "part": "<部位>", "id": <编号或 null>},
    ...
  ]
}

现在，请读取给定图片与 targets，按上述规则为每个目标返回编号，仅输出符合格式的 JSON。
"""



class JSONLLogger:
    """
    以 JSON Lines 方式写日志：一行一个事件
    - 默认不写入 base64/image data url（只写长度/摘要）
    - 可配置文本截断长度，避免日志过大
    """
    def __init__(self, log_path: Optional[str], *, max_text_len: int = 4000, include_base64: bool = False):
        self.log_path = log_path
        self.max_text_len = max_text_len
        self.include_base64 = include_base64
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if os.path.dirname(self.log_path) else None

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _truncate(self, s: Any) -> Any:
        if s is None:
            return None
        if isinstance(s, str):
            if len(s) <= self.max_text_len:
                return s
            return s[: self.max_text_len] + f"...(truncated,len={len(s)})"
        return s

    def _summarize_big_str(self, s: Optional[str]) -> Any:
        if not s:
            return None
        if self.include_base64:
            return self._truncate(s)
        h = hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return {"len": len(s), "md5_8": h, "note": "omitted"}

    def _safe_obj(self, obj: Any) -> Any:
        try:
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    lk = str(k).lower()
                    if isinstance(v, str) and ("base64" in lk or "data_url" in lk):
                        out[k] = self._summarize_big_str(v)
                    else:
                        out[k] = self._safe_obj(v)
                return out
            if isinstance(obj, list):
                return [self._safe_obj(x) for x in obj]
            if isinstance(obj, str):
                return self._truncate(obj)
            if isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, np.ndarray):
                return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
            if isinstance(obj, Image.Image):
                return {"type": "PIL.Image", "size": list(obj.size), "mode": obj.mode}
            return self._truncate(str(obj))
        except Exception:
            return self._truncate(str(obj))

    def log(self, step: str, *, inp: Any = None, out: Any = None, extra: Any = None):
        if not self.log_path:
            return
        evt = {
            "ts": self._now(),
            "step": step,
            "input": self._safe_obj(inp),
            "output": self._safe_obj(out),
            "extra": self._safe_obj(extra),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")


# ============================================================
# 🧩 工具函数：图像加载/编码
# ============================================================

def load_image_any(image_path_or_ndarray: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    """支持 str路径 / ndarray(BGR or RGB) / PIL.Image"""
    if isinstance(image_path_or_ndarray, Image.Image):
        return image_path_or_ndarray.convert("RGB")

    if isinstance(image_path_or_ndarray, np.ndarray):
        arr = image_path_or_ndarray
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                rgb = arr[..., ::-1].copy()  # BGR->RGB
                return Image.fromarray(rgb).convert("RGB")
            if arr.shape[2] == 4:
                return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"Unsupported ndarray shape: {arr.shape}")

    if isinstance(image_path_or_ndarray, str):
        return Image.open(image_path_or_ndarray).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_path_or_ndarray)}")


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_data_url(image_path_or_ndarray: Union[str, np.ndarray, Image.Image]) -> str:
    """返回 data:image/jpeg;base64,... 用于 Azure Vision"""
    img = load_image_any(image_path_or_ndarray)
    b = pil_to_jpeg_bytes(img)
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def ensure_base64_no_header(data_url_or_b64: str) -> str:
    """后端若需要纯 base64，这里去掉 data: 头"""
    if data_url_or_b64.startswith("data:"):
        return data_url_or_b64.split(",", 1)[1]
    return data_url_or_b64


# ============================================================
# 🧠 Azure 调用（Chat Completions + image_url） + 日志
# ============================================================

def call_azure_chat(
    system_prompt: str,
    user_text: str,
    image_data_url: Optional[str],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: int = 60,
    logger: Optional[JSONLLogger] = None,
    log_step: str = "azure_chat",
) -> str:
    if not AZURE_KEY:
        raise RuntimeError("AZURE_KEY is empty. Please set env var AZURE_KEY.")

    url = (
        f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_API_VERSION}"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    if image_data_url:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_data_url, "detail": "high"}
        })

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if logger:
        logger.log(
            log_step + ".request",
            inp={
                "url": url,
                "deployment": AZURE_DEPLOYMENT,
                "api_version": AZURE_API_VERSION,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "user_text": user_text,
                "system_prompt_len": len(system_prompt),
                "has_image": bool(image_data_url),
                "image_data_url": image_data_url,  # logger 会摘要
            },
        )

    r = requests.post(
        url,
        headers={"Content-Type": "application/json", "api-key": AZURE_KEY},
        json=payload,
        timeout=timeout,
    )

    if not r.ok:
        if logger:
            logger.log(log_step + ".error", inp={"status_code": r.status_code}, out={"text": r.text})
        raise RuntimeError(f"Azure API Error ({r.status_code}): {r.text}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]

    if logger:
        logger.log(
            log_step + ".response",
            out={
                "status_code": r.status_code,
                "finish_reason": data["choices"][0].get("finish_reason", None),
                "content": content,
            },
        )

    return content


def parse_json_strict(s: str) -> Any:
    """兼容 ```json ... ``` 包裹"""
    s2 = s.strip()
    s2 = re.sub(r"^\s*```json\s*", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```\s*$", "", s2)
    return json.loads(s2)


# ============================================================
# 📡 后端接口：/vision/sam, /vision/dino + 日志
# ============================================================

def backend_post_sam(
    image_obj: Union[str, np.ndarray, Image.Image],
    min_area: int = 1,
    timeout: int = 120,
    logger: Optional[JSONLLogger] = None,
) -> Dict[str, Any]:
    img = load_image_any(image_obj)
    img_bytes = pil_to_jpeg_bytes(img)

    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    data = {"min_area": str(min_area)}

    if logger:
        logger.log("backend.sam.request", inp={"url": f"{BACKEND_URL}/vision/sam", "min_area": min_area, "image": img})

    r = requests.post(f"{BACKEND_URL}/vision/sam", files=files, data=data, timeout=timeout)
    if not r.ok:
        if logger:
            logger.log("backend.sam.error", inp={"status_code": r.status_code}, out={"text": r.text})
        raise RuntimeError(f"Backend /vision/sam Error ({r.status_code}): {r.text}")

    out = r.json()

    if logger:
        logger.log(
            "backend.sam.response",
            out={
                "keys": list(out.keys()),
                "npz_path": out.get("npz_path"),
                "has_vis": bool(out.get("centers_vis_base64") or out.get("sam_vis_base64")),
            },
        )

    return out


def backend_post_dino(
    image_obj: Union[str, np.ndarray, Image.Image],
    npz_path: str,
    target_ids: List[int],
    timeout: int = 180,
    logger: Optional[JSONLLogger] = None,
) -> Dict[str, Any]:
    img = load_image_any(image_obj)
    img_bytes = pil_to_jpeg_bytes(img)

    files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
    data = {
        "npz_path": npz_path,
        "target_ids": ",".join(str(i) for i in target_ids),
    }

    if logger:
        logger.log("backend.dino.request", inp={"url": f"{BACKEND_URL}/vision/dino", "npz_path": npz_path, "target_ids": target_ids, "image": img})

    r = requests.post(f"{BACKEND_URL}/vision/dino", files=files, data=data, timeout=timeout)
    if not r.ok:
        if logger:
            logger.log("backend.dino.error", inp={"status_code": r.status_code}, out={"text": r.text})
        raise RuntimeError(f"Backend /vision/dino Error ({r.status_code}): {r.text}")

    out = r.json()

    if logger:
        kps = out.get("keypoints", []) or []
        logger.log(
            "backend.dino.response",
            out={
                "keys": list(out.keys()),
                "keypoints_count": len(kps),
                "has_vis": bool(out.get("dino_vis_base64") or out.get("vis_base64")),
            },
        )

    return out


# ============================================================
# 🔧 坐标与合并逻辑
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def maybe_pixel_from_keypoint_item(kp: Dict[str, Any]) -> Optional[List[float]]:
    if "x" not in kp or "y" not in kp:
        return None
    return [float(kp["x"]), float(kp["y"])]


def normalize_keypoints_to_pixel(keypoints: List[Dict[str, Any]], W: int, H: int) -> Dict[int, List[int]]:
    """
    把 keypoints 转成 {id: [u,v]} 像素字典
    规则：
    - 若 x<=1.5 且 y<=1.5，认为是 0~1 归一化
    - 否则认为是像素
    """
    out: Dict[int, List[int]] = {}
    for kp in keypoints:
        if "id" not in kp:
            continue
        pid = int(kp["id"])
        xy = maybe_pixel_from_keypoint_item(kp)
        if not xy:
            continue
        x, y = xy
        if x <= 1.5 and y <= 1.5:
            u = int(round(clamp(x, 0.0, 1.0) * (W - 1)))
            v = int(round(clamp(y, 0.0, 1.0) * (H - 1)))
        else:
            u = int(round(clamp(x, 0.0, W - 1)))
            v = int(round(clamp(y, 0.0, H - 1)))
        out[pid] = [u, v]
    return out


def extract_target_ids_from_stage1(stage1_json: Dict[str, Any]) -> List[int]:
    ids: List[int] = []
    for obj in stage1_json.get("objects", []) or []:
        for comp in obj.get("components", []) or []:
            cid = comp.get("id", None)
            if isinstance(cid, int):
                ids.append(cid)
            elif isinstance(cid, float) and cid.is_integer():
                ids.append(int(cid))

    # 去重但保持稳定顺序
    seen = set()
    uniq = []
    for i in ids:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def extract_targets_from_action_plan(stage2_json: Dict[str, Any]) -> List[List[str]]:
    """
    Stage3 需要 targets = [[obj, part], ...]
    从 action_components 抽 component，并过滤 arm/robot
    """
    targets: List[List[str]] = []
    for item in stage2_json.get("action_components", []) or []:
        comp = item.get("component", None)
        if not (isinstance(comp, list) and len(comp) == 2):
            continue
        obj = str(comp[0]).strip()
        part = str(comp[1]).strip()
        if obj.lower() in ("arm", "robot"):
            continue
        targets.append([obj, part])

    # 去重（保持顺序）
    seen = set()
    out = []
    for t in targets:
        key = (t[0].lower().strip(), t[1].lower().strip())
        if key not in seen:
            out.append(t)
            seen.add(key)
    return out


def build_map_dict(stage3_json: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """
    {"object":"..","part":"..","id": 12/null} -> dict["obj|part"]=id
    """
    m: Dict[str, Optional[int]] = {}
    for r in stage3_json.get("results", []) or []:
        obj = str(r.get("object", "")).lower().strip()
        part = str(r.get("part", "")).lower().strip()
        rid = r.get("id", None)
        if rid is None:
            m[f"{obj}|{part}"] = None
        else:
            try:
                m[f"{obj}|{part}"] = int(rid)
            except Exception:
                m[f"{obj}|{part}"] = None
    return m


def write_pixels_back(
    stage2_json: Dict[str, Any],
    map_dict: Dict[str, Optional[int]],
    id_to_pixel: Dict[int, List[int]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    把每个 action_components 里非 arm/robot 的动作写回 args.pixel
    返回：(merged_plan_json, debug_detections)
    """
    merged = json.loads(json.dumps(stage2_json, ensure_ascii=False))  # deep copy
    debug: Dict[str, Any] = {"items": []}

    for act in merged.get("action_components", []) or []:
        comp = act.get("component", None)
        if not (isinstance(comp, list) and len(comp) == 2):
            continue
        obj = str(comp[0]).lower().strip()
        part = str(comp[1]).lower().strip()
        if obj in ("arm", "robot"):
            continue

        key = f"{obj}|{part}"
        rid = map_dict.get(key, None)

        rec: Dict[str, Any] = {
            "object": obj,
            "part": part,
            "mapped_id": rid,
            "pixel": None,
            "error": None,
        }

        if rid is None:
            rec["error"] = "No matching ID found (null)"
            act.setdefault("args", {})
            act["args"].setdefault("pixel", None)
        else:
            px = id_to_pixel.get(int(rid), None)
            if px is None:
                rec["error"] = f"Found ID {rid} but no keypoint coords"
                act.setdefault("args", {})
                act["args"]["pixel"] = None
            else:
                rec["pixel"] = px
                act.setdefault("args", {})
                act["args"]["pixel"] = px

        debug["items"].append(rec)

    return merged, debug


# ============================================================
# ✅ 核心入口：与旧版对齐 + 每步日志
# ============================================================

def plan_and_ground_robot_actions(
    instruction: str,
    image_path: Union[str, np.ndarray, Image.Image],
    save_vis_prefix: str = "logs/run1",
    *,
    min_area: int = 1,
    azure_timeout: int = 60,
    backend_timeout_sam: int = 120,
    backend_timeout_dino: int = 180,
    # 日志参数
    log_path: Optional[str] = None,
    log_max_text_len: int = 4000,
    log_include_base64: bool = False,
) -> Dict[str, Any]:
    """
    新版 v2 管线：
    1) /vision/sam -> 得到 npz_path, centers_vis_base64
    2) LLM Stage1(基于 sam 可视化图) -> 提取 target_ids
    3) /vision/dino(image + npz + target_ids) -> keypoints + dino_vis_base64
    4) LLM Stage2(基于原图) -> action plan
    5) LLM Stage3(基于 dino 可视化图 + targets) -> id mapping
    6) 写回 action_components[*].args.pixel = [u,v]
    7) 返回 action_json_str/detections/vis_path
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(save_vis_prefix), exist_ok=True) if os.path.dirname(save_vis_prefix) else None

    logger = JSONLLogger(log_path, max_text_len=log_max_text_len, include_base64=log_include_base64) if log_path else None
    if logger:
        logger.log("start", inp={"instruction": instruction, "save_vis_prefix": save_vis_prefix, "min_area": min_area})

    # 准备原图信息
    orig_img = load_image_any(image_path)
    W, H = orig_img.size

    raw_image_data_url = to_data_url(orig_img)
    if logger:
        logger.log("image.loaded", inp={"image": orig_img}, out={"W": W, "H": H, "raw_image_data_url": raw_image_data_url})

    # 1) 后端 SAM
    sam_res = backend_post_sam(orig_img, min_area=min_area, timeout=backend_timeout_sam, logger=logger)
    npz_path = sam_res.get("npz_path", None)
    sam_vis_base64 = sam_res.get("centers_vis_base64", None) or sam_res.get("sam_vis_base64", None)

    if not npz_path:
        if logger:
            logger.log("sam.missing_npz_path", out={"sam_res_keys": list(sam_res.keys())})
        raise RuntimeError(f"/vision/sam response missing npz_path: {sam_res.keys()}")

    sam_vis_data_url = None
    if sam_vis_base64:
        sam_vis_data_url = sam_vis_base64 if str(sam_vis_base64).startswith("data:") else f"data:image/jpeg;base64,{sam_vis_base64}"

    if logger:
        logger.log("sam.parsed", out={"npz_path": npz_path, "sam_vis_data_url": sam_vis_data_url})

    # 2) Stage1：意图 + 收集编号
    stage1_str = call_azure_chat(
        system_prompt=PROMPT_STAGE_1,
        user_text=instruction,
        image_data_url=sam_vis_data_url,
        timeout=azure_timeout,
        temperature=0.0,
        logger=logger,
        log_step="stage1",
    )
    stage1_json = parse_json_strict(stage1_str)
    target_ids = extract_target_ids_from_stage1(stage1_json)

    if logger:
        logger.log("stage1.parsed", inp={"stage1_str": stage1_str}, out={"stage1_json": stage1_json, "target_ids": target_ids})

    # 3) 后端 DINO（只对 target_ids 做）
    dino_res = backend_post_dino(orig_img, npz_path=npz_path, target_ids=target_ids, timeout=backend_timeout_dino, logger=logger)
    keypoints = dino_res.get("keypoints", []) or []
    dino_vis_base64 = dino_res.get("dino_vis_base64", None) or dino_res.get("vis_base64", None)

    dino_vis_data_url = None
    if dino_vis_base64:
        dino_vis_data_url = dino_vis_base64 if str(dino_vis_base64).startswith("data:") else f"data:image/jpeg;base64,{dino_vis_base64}"

    id_to_pixel = normalize_keypoints_to_pixel(keypoints, W=W, H=H)

    if logger:
        logger.log("dino.parsed", out={"keypoints_count": len(keypoints), "dino_vis_data_url": dino_vis_data_url, "id_to_pixel": id_to_pixel})

    # 4) Stage2：规划动作
    stage2_str = call_azure_chat(
        system_prompt=PROMPT_STAGE_2,
        user_text=instruction,
        image_data_url=raw_image_data_url,
        timeout=azure_timeout,
        temperature=0.0,
        logger=logger,
        log_step="stage2",
    )
    stage2_json = parse_json_strict(stage2_str)

    if logger:
        logger.log("stage2.parsed", inp={"stage2_str": stage2_str}, out={"stage2_json": stage2_json})

    # 5) Stage3：targets -> id 映射（用 dino 可视化图）
    targets = extract_targets_from_action_plan(stage2_json)
    stage3_user_text = f"targets = {json.dumps(targets, ensure_ascii=False)}"

    stage3_str = call_azure_chat(
        system_prompt=PROMPT_STAGE_3_TEMPLATE,
        user_text=stage3_user_text,
        image_data_url=dino_vis_data_url,
        timeout=azure_timeout,
        temperature=0.0,
        logger=logger,
        log_step="stage3",
    )
    stage3_json = parse_json_strict(stage3_str)
    map_dict = build_map_dict(stage3_json)

    if logger:
        logger.log("stage3.parsed", inp={"stage3_str": stage3_str, "targets": targets}, out={"stage3_json": stage3_json, "map_dict": map_dict})

    # 6) 写回 args.pixel
    merged_plan, debug_det = write_pixels_back(stage2_json, map_dict, id_to_pixel)

    if logger:
        logger.log("merge.write_pixels_back", out={"merged_plan": merged_plan, "debug_det": debug_det})

    # 7) 输出对齐旧版字段
    out_action_str = json.dumps(merged_plan, ensure_ascii=False)

    # 可视化路径：把 dino 可视化写文件（可选）
    vis_path = None
    try:
        if dino_vis_base64:
            vis_path = f"{save_vis_prefix}_dino_vis.jpg"
            raw_b64 = ensure_base64_no_header(str(dino_vis_base64))
            with open(vis_path, "wb") as f:
                f.write(base64.b64decode(raw_b64))
    except Exception as e:
        if logger:
            logger.log("vis.save.error", out={"error": str(e)})
        vis_path = None

    elapsed = round(time.time() - t0, 3)

    res: Dict[str, Any] = {
        "action_json_str": out_action_str,
        "detections": {
            "stage1": stage1_json,
            "target_ids": target_ids,
            "stage2": stage2_json,
            "stage3": stage3_json,
            "id_to_pixel": id_to_pixel,
            "debug": debug_det,
            "backend": {
                "sam": sam_res,
                "dino": {k: v for k, v in dino_res.items() if k != "dino_vis_base64"},
            },
            "meta": {"W": W, "H": H, "elapsed_sec": elapsed},
        },
        "vis_path": vis_path,
    }

    if logger:
        logger.log("done", out={"vis_path": vis_path, "elapsed_sec": elapsed, "action_json_str_len": len(out_action_str)})

    return res


# ============================================================
# 🧪 本地测试入口（可选）
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="image path")
    parser.add_argument("--ins", required=True, help="instruction text")
    parser.add_argument("--save", default="logs/run1", help="save_vis_prefix")

    # 日志相关
    parser.add_argument("--log", default=None, help="log file path (jsonl). e.g. logs/run1.jsonl")
    parser.add_argument("--log_max_text", type=int, default=4000, help="max text length in log")
    parser.add_argument("--log_include_base64", action="store_true", help="include base64/data_url in log (huge!)")

    # 其他参数（可选）
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--azure_timeout", type=int, default=60)
    parser.add_argument("--sam_timeout", type=int, default=120)
    parser.add_argument("--dino_timeout", type=int, default=180)

    args = parser.parse_args()

    out = plan_and_ground_robot_actions(
        instruction=args.ins,
        image_path=args.img,
        save_vis_prefix=args.save,
        min_area=args.min_area,
        azure_timeout=args.azure_timeout,
        backend_timeout_sam=args.sam_timeout,
        backend_timeout_dino=args.dino_timeout,
        log_path=args.log,
        log_max_text_len=args.log_max_text,
        log_include_base64=args.log_include_base64,
    )

    print("=== action_json_str ===")
    print(out["action_json_str"])
    print("=== vis_path ===", out["vis_path"])
    if args.log:
        print("=== log_path ===", args.log)
