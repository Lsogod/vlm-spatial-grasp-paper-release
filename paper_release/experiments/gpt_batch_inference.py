# -*- coding: utf-8 -*-
import os
import json
import glob
import time
import base64
import argparse
import threading
import re
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import AzureOpenAI  # ⚠️ 修改1：导入 AzureOpenAI
import traceback
from PIL import Image, ImageDraw, ImageFont
import textwrap

# ============================================================
# ⚙️ Azure OpenAI 配置区域
# ============================================================
# ⚠️ 使用环境变量，避免在代码中存储密钥
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-5-chat")

# 注意：Azure 上目前最好的视觉模型是 GPT-4o 或 GPT-4 Turbo with Vision
# Qwen 模型通常不在 Azure OpenAI 官方服务中，除非你用的是 MaaS (Model as a Service) 
# 如果你确定是在用 Azure 调 Qwen，请确保你的 Endpoint 是支持 OpenAI 协议的 MaaS 端点

DEFAULT_WORKERS = 4

# ============================================================
# 🧠 Step1: 只输出动作序列（不输出坐标/bbox）
# ============================================================

PROMPT_STEP_1_PLAN = r"""
# Role
你是一个具身智能任务规划助手。你需要根据【图片】和【指令】生成机械臂动作序列。

# Key Requirement (Very Important)
- **本步骤只输出动作序列与操作对象（component）和描述（description）**
- **绝对不要输出任何坐标、bbox、point、像素位置**
- 后续会有第二步专门做定位。

# Action Set (固定动作集合)
只能从以下动作中选择：
- move_to_grasp_pose
- close_gripper
- move_to_point
- open_gripper
- back_to_zero

# Component Format
每个动作必须包含 component：
- component = ["<英文物体名>", "<英文部位名>"]
- 物体名必须英文（如 peach / banana / mouse / bowl / blue bowl）
- 部位名必须英文，尽量使用这些规范词：
  - center / middle / top / bottom / handle / opening

# Task Logic
- grasp_only：
  move_to_grasp_pose -> close_gripper -> back_to_zero
- pick_and_place：
  move_to_grasp_pose -> close_gripper -> move_to_point -> open_gripper -> back_to_zero

# IMPORTANT: component 规则（确保第二步能定位）
- move_to_grasp_pose / close_gripper：component 指向【被抓取物体】及其部位（例如 ["banana","middle"]）
- move_to_point / open_gripper：component 指向【放置目标容器/位置】及其部位（例如 ["blue bowl","center"]）
- back_to_zero：component 固定为 ["arm","base"]

# Output Format (Strict JSON Only)
仅输出一个 JSON 对象：
{
  "action_components": [
    {
      "component": ["object_name_en", "part_name_en"],
      "action": "one_action_from_action_set",
      "description": "中文简述"
    }
  ]
}
不要输出其它任何多余文本。
"""

# ============================================================
# 🧠 Step2: 定位 component 的中心点（输出 point_2d）
# 注意：这里必须用 {{ }} 避免 .format() KeyError
# ============================================================

PROMPT_STEP_2_LOCATE = r"""
# Role
You are a precise visual localization assistant.

# Task
Locate the **geometric center point** of the target described as:
**{object_name} {part_name}**.

# Part meaning
- center: geometric center of the object
- middle: midpoint along the main body axis (for elongated objects) or the center of the object body
- opening: geometric center of the visible opening interior (for bowls/cups)
- handle: center point of the handle region
- top/bottom: center point of the top/bottom region

# Disambiguation
If multiple instances exist, choose the one that best matches the description and is most central.

# Output Format
Return a single JSON object. Coordinates are RELATIVE integers in [0, 1000].
{{
  "point_2d": [x, y]
}}
Return JSON only.
"""

# ============================================================
# 🛠️ 工具函数
# ============================================================

def np_image_to_data_url(img_rgb: np.ndarray) -> str:
    """np RGB -> data:image/jpeg;base64,..."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise ValueError("Could not encode image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def clean_json(text: str) -> str:
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return text

def call_model(client: AzureOpenAI, messages, max_retries=5):
    """
    ⚠️ 修改2：Azure 调用逻辑
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME, # Azure 中 model 参数通常传 deployment name
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)

            # 429 限流：指数退避
            if "429" in msg or "RateLimit" in msg:
                wait = min(60, 2 ** attempt)
                print(f"⏳ [429] Rate limited, sleep {wait}s ...")
                time.sleep(wait)
                continue

            print(f"❌ [API] attempt {attempt+1}/{max_retries} failed: {type(e).__name__}: {e}")
            time.sleep(1 + attempt)

    return None

def need_locate(component):
    if not isinstance(component, list) or len(component) != 2:
        return False
    obj, part = component
    if str(obj).strip() == "arm" and str(part).strip() == "base":
        return False
    return True

def step_2_locate(client: AzureOpenAI, img_data_url: str, object_name: str, part_name: str, width: int, height: int):
    prompt = PROMPT_STEP_2_LOCATE.format(object_name=object_name, part_name=part_name)
    query_text = f"{object_name} {part_name}".strip()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_data_url}},
            {"type": "text", "text": f"Locate: {query_text}"}
        ]}
    ]

    raw = call_model(client, messages)
    if not raw:
        return None

    try:
        data = json.loads(clean_json(raw))
        pt = data.get("point_2d")
        if pt and isinstance(pt, list) and len(pt) == 2:
            x_px = int((pt[0] / 1000.0) * width)
            y_px = int((pt[1] / 1000.0) * height)
            x_px = max(0, min(width, x_px))
            y_px = max(0, min(height, y_px))
            return [x_px, y_px]
    except Exception:
        pass

    return None

# ============================================================
# 🎨 可视化（保持不变）
# ============================================================

def draw_text_with_pillow(img_bgr, text, position, font_size=22, bg_opacity=220):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    font = None
    # 尝试加载常用中文字体
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except:
                pass
    if font is None:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox(position, text, font=font)
    except Exception:
        w, h = draw.textsize(text, font=font)
        bbox = (position[0], position[1], position[0] + w, position[1] + h)

    if bg_opacity > 0:
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle(bbox, fill=(0, 0, 0, bg_opacity))
        pil_img = Image.alpha_composite(pil_img, overlay)
        draw = ImageDraw.Draw(pil_img)

    draw.text(position, text, font=font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def draw_visualization(img_rgb, actions, save_path, instruction=None, task_index=None, base_name=None):
    if img_rgb is None:
        print(f"❌ [Vis] Image is None, skipping save: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    y = 10
    if instruction:
        header = f"指令: {instruction}"
        if base_name is not None and task_index is not None:
            header = f"{base_name} | Task {task_index} | 指令: {instruction}"

        for line in textwrap.wrap(header, width=28):
            img_bgr = draw_text_with_pillow(img_bgr, line, (10, y), font_size=24, bg_opacity=220)
            y += 34
        y += 6

    for i, act in enumerate(actions):
        action_name = act.get("action", "unknown")
        comp = act.get("component", ["N/A", "N/A"])
        comp_str = f"{comp[0]}-{comp[1]}" if isinstance(comp, list) and len(comp) == 2 else str(comp)
        line = f"[{i+1}] {action_name} ({comp_str})"
        img_bgr = draw_text_with_pillow(img_bgr, line, (10, y), font_size=18, bg_opacity=160)
        y += 26

    drawn_count = 0
    for i, act in enumerate(actions):
        name = act.get("action", "unknown")
        pt = act.get("coords")

        if pt and isinstance(pt, list) and len(pt) == 2 and all(isinstance(n, (int, float)) for n in pt):
            cx, cy = int(pt[0]), int(pt[1])
            is_grasp = ("grasp" in name) or ("close" in name)
            color = (0, 0, 255) if is_grasp else (255, 0, 0)

            cv2.circle(img_bgr, (cx, cy), 8, color, -1)
            cv2.circle(img_bgr, (cx, cy), 10, (255, 255, 255), 2)

            label = f"[{i+1}] {name}"
            offset_y = -20 if i % 2 == 0 else 25
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_bgr, (cx + 15, cy + offset_y - th - 5), (cx + 15 + tw, cy + offset_y + 5), (0, 0, 0), -1)
            cv2.putText(img_bgr, label, (cx + 15, cy + offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            drawn_count += 1

    ok = cv2.imwrite(save_path, img_bgr)
    if ok:
        print(f"🖼️ [Vis] Saved with {drawn_count} points: {save_path}")
    else:
        print(f"❌ [Vis] cv2.imwrite failed for: {save_path}")

# ============================================================
# 🚀 推理核心
# ============================================================

def run_inference_pipeline(client: AzureOpenAI, image_rgb: np.ndarray, instruction: str):
    h, w = image_rgb.shape[:2]
    img_data_url = np_image_to_data_url(image_rgb)

    # ---------- Step 1: Plan ----------
    msgs_1 = [
        {"role": "system", "content": PROMPT_STEP_1_PLAN},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_data_url}},
            {"type": "text", "text": f"Instruction: {instruction}"}
        ]}
    ]

    raw_plan = call_model(client, msgs_1)
    if not raw_plan:
        return {"status": "failed", "error": "Step 1 API failed"}

    try:
        plan_data = json.loads(clean_json(raw_plan))
        action_components = plan_data.get("action_components", [])
        if not isinstance(action_components, list) or len(action_components) == 0:
            return {"status": "failed", "error": "Step 1 returned empty action_components", "raw": raw_plan}
    except Exception as e:
        return {"status": "failed", "error": f"Step 1 JSON error: {e}", "raw": raw_plan}

    # ---------- Step 2: Locate ----------
    comp_map = {}
    next_id = 0
    unique = []
    seen = set()
    for act in action_components:
        comp = act.get("component")
        if need_locate(comp):
            key = (str(comp[0]).strip(), str(comp[1]).strip())
            if key not in seen:
                seen.add(key)
                unique.append(key)

    for (obj, part) in unique:
        coords = step_2_locate(client, img_data_url, obj, part, w, h)
        if coords is None:
            return {"status": "failed", "error": f"Step 2 locate failed for component: {obj} {part}"}
        comp_map[(obj, part)] = {"target_id": next_id, "coords": coords}
        next_id += 1

    # ---------- Merge ----------
    merged = []
    for act in action_components:
        comp = act.get("component")
        out = {
            "component": comp,
            "action": act.get("action"),
            "description": act.get("description", ""),
            "target_id": None,
            "coords": None
        }

        if need_locate(comp):
            key = (str(comp[0]).strip(), str(comp[1]).strip())
            if key in comp_map:
                out["target_id"] = comp_map[key]["target_id"]
                out["coords"] = comp_map[key]["coords"]

        merged.append(out)

    return {
        "status": "success",
        "instruction": instruction,
        "image_size": [w, h],
        "action_components": merged
    }

# ============================================================
# 📂 主处理逻辑
# ============================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def upsert_result(results_list, new_item):
    idx = new_item.get("task_index")
    for i, it in enumerate(results_list):
        if it.get("task_index") == idx:
            results_list[i] = new_item
            return
    results_list.append(new_item)

def process_single_npz(client: AzureOpenAI, npz_path: str, output_dir: str, img_output_dir: str):
    thread_name = threading.current_thread().name
    base_name = os.path.basename(npz_path).replace(".npz", "")
    json_path = os.path.join(output_dir, f"{base_name}.json")

    try:
        # 1. 读取 NPZ 数据
        data = np.load(npz_path, allow_pickle=True)
        if "image" not in data.files:
            return
        image_rgb = data["image"]

        trajectories = []
        if "trajectories" in data.files:
            raw = data["trajectories"]
            s_traj = str(raw.item()) if getattr(raw, "ndim", 0) == 0 else str(raw)
            try:
                trajectories = json.loads(s_traj)
            except:
                trajectories = []

        if not trajectories:
            return

        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # 2. 加载 JSON 并执行清洗逻辑
        final_output = {}
        processed_indices = set()

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    final_output = json.load(f)

                # =======================================================
                # 🛠️ 核心修改：加载时清洗数据
                # 只保留成功的记录，丢弃失败的记录以便重试
                # =======================================================
                valid_results = []
                existing_results = final_output.get("results", [])

                for res in existing_results:
                    status = res.get("ai_inference", {}).get("status")
                    t_idx = res.get("task_index")

                    if status == "success":
                        # 成功的任务：保留并在本次运行中跳过
                        valid_results.append(res)
                        if t_idx is not None:
                            processed_indices.add(t_idx)
                    else:
                        # 失败的任务：被丢弃 (drop)，不加入 processed_indices
                        # 这样后续循环会检测到它未完成，自动重试
                        err = res.get("ai_inference", {}).get("error", "Unknown")
                        # print(f"[{thread_name}] ⚠️ Dropping failed history Task {t_idx}. Will Retry.")
                
                # 用清洗后的列表覆盖原列表
                final_output["results"] = valid_results
                # =======================================================

            except Exception as e:
                print(f"[{thread_name}] ⚠️ JSON load error {json_path}: {e}")
                final_output = {"source_file": npz_path, "results": []}
        else:
            final_output = {"source_file": npz_path, "results": []}

        if "results" not in final_output:
            final_output["results"] = []

        # 3. 遍历任务进行推理
        any_updates = False

        for idx, traj in enumerate(trajectories):
            # === 🟢 跳过已完成任务 ===
            if idx in processed_indices:
                continue

            instruction = (traj.get("instruction", "") or "").strip()
            if not instruction:
                continue

            print(f"[{thread_name}] 🔹 Running {base_name} [Task {idx}]: {instruction}")

            # 执行推理
            result = run_inference_pipeline(client, image_rgb, instruction)

            # 可视化 (成功才保存)
            if result.get("status") == "success":
                vis_filename = f"{base_name}_task{idx}.jpg"
                vis_path = os.path.join(img_output_dir, vis_filename)

                draw_visualization(
                    image_rgb,
                    result.get("action_components", []),
                    vis_path,
                    instruction=instruction,
                    task_index=idx,
                    base_name=base_name
                )
            else:
                print(f"[{thread_name}] ⚠️ Inference Failed: {result.get('error')}")

            # 构造新记录
            new_item = {
                "task_index": idx,
                "original_instruction": instruction,
                "ai_inference": result
            }

            # 更新结果列表 (upsert_result 负责处理同ID覆盖)
            upsert_result(final_output["results"], new_item)
            any_updates = True

            # 实时保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, cls=NpEncoder, indent=2, ensure_ascii=False)

        if any_updates:
            print(f"[{thread_name}] ✅ Updated: {base_name}.json")

    except Exception as e:
        traceback.print_exc()
        print(f"[{thread_name}] ❌ Error in {base_name}: {e}")
def worker_task(args):
    # ⚠️ 修改3：Worker 中初始化 AzureClient
    npz_path, output_dir, img_output_dir = args
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )
    process_single_npz(client, npz_path, output_dir, img_output_dir)

def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI Batch Inference")
    parser.add_argument("--input_dir", type=str, default="../dist/all", help="Input .npz folder")
    parser.add_argument("--output_dir", type=str, default="gpt-5_results", help="JSON output folder")
    parser.add_argument("--img_output_dir", type=str, default=None, help="Image output folder (Optional)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise RuntimeError("Missing Azure config. Please set AZURE_ENDPOINT and AZURE_API_KEY.")

    os.makedirs(args.output_dir, exist_ok=True)

    img_out = args.img_output_dir if args.img_output_dir else os.path.join(args.output_dir, "vis_images")
    os.makedirs(img_out, exist_ok=True)

    npz_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
    print(f"🔍 Found {len(npz_files)} files. Saving images to: {img_out}")

    # ⚠️ 修改4：不再传递 API_KEY / BASE_URL，改在 worker 内部用全局变量初始化
    task_args = [(f, args.output_dir, img_out) for f in npz_files]

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = [executor.submit(worker_task, arg) for arg in task_args]

    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                future.result()
            except Exception as exc:
                print(f"\n❌ Thread generated an exception: {exc}")
    except KeyboardInterrupt:
        print("\n\n🛑 Detected Ctrl+C! Forcing stop...")
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(0)
    finally:
        executor.shutdown(wait=True)

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
