# -*- coding: utf-8 -*-
import os
import sys 
import json
import glob
import time
import base64
import argparse
import requests
import numpy as np
import cv2
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# ⚙️ 配置区域 (请修改这里)
# ============================================================

# 视觉后端地址
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Azure OpenAI 配置
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5-chat")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_KEY = os.getenv("AZURE_API_KEY", "")


# ============================================================
# 🛑 核心提示词 (PROMPTS) - 完全保留
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


# ============================================================
# 🛠️ 核心工具函数
# ============================================================

def np_image_to_bytes(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode('.jpg', img_bgr)
    if not success: raise ValueError("Could not encode image")
    return encoded_image.tobytes()

def np_image_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode('.jpg', img_bgr)
    if not success: raise ValueError("Could not encode image")
    b64_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def save_base64_image(b64_str, save_path):
    try:
        if "," in b64_str: b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f: f.write(img_data)
    except Exception as e:
        print(f"❌ Failed to save image {save_path}: {e}")

def parse_json_gpt(text):
    try:
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            clean = "\n".join(lines)
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}

def call_azure(system_prompt, user_text, img_base64=None):
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_KEY}
    content = [{"type": "text", "text": user_text}]
    if img_base64:
        content.append({"type": "image_url", "image_url": {"url": img_base64, "detail": "high"}})
    payload = {
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 2048
    }
    
    # === 增强重试逻辑 ===
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # 超时时间设为 300s，防止复杂图片超时
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            
            if resp.status_code == 429:
                wait_time = 5 * (attempt + 1) # 指数退避
                print(f"⚠️ Azure 429 Rate Limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            if 500 <= resp.status_code < 600:
                time.sleep(5)
                continue

            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            print(f"⚠️ Azure Timeout. Retrying ({attempt+1}/{max_retries})...")
            time.sleep(5)
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2)
            else: return "{}"
    return "{}"


# ============================================================
# 🚀 完整推理流程 (Pipeline Logic)
# ============================================================

def run_inference_pipeline(image_rgb, instruction):
    try:
        img_bytes = np_image_to_bytes(image_rgb)
        img_b64_raw = np_image_to_base64(image_rgb)

        # 1. SAM
        files = {'image': ('img.jpg', img_bytes, 'image/jpeg')}
        resp_sam = requests.post(f"{BACKEND_URL}/vision/sam", files=files, data={'min_area': 10})
        resp_sam.raise_for_status()
        res_sam = resp_sam.json()
        npz_path = res_sam['npz_path']
        sam_vis_b64_raw = res_sam['centers_vis_base64']
        
        sam_vis_b64_azure = sam_vis_b64_raw
        if not sam_vis_b64_azure.startswith("data:"):
            sam_vis_b64_azure = f"data:image/jpeg;base64,{sam_vis_b64_azure}"

        # 2. GPT Intent
        gpt_res1 = call_azure(PROMPT_STAGE_1, instruction, sam_vis_b64_azure)
        json_res1 = parse_json_gpt(gpt_res1)
        target_ids = []
        for obj in json_res1.get("objects", []):
            for comp in obj.get("components", []):
                if isinstance(comp.get("id"), int):
                    target_ids.append(comp["id"])
        
        # 3. DINO
        files_dino = {'image': ('img.jpg', img_bytes, 'image/jpeg')}
        data_dino = {'npz_path': npz_path, 'target_ids': ",".join(map(str, target_ids))}
        resp_dino = requests.post(f"{BACKEND_URL}/vision/dino", files=files_dino, data=data_dino)
        resp_dino.raise_for_status()
        res_dino = resp_dino.json()
        keypoints = res_dino['keypoints']
        dino_vis_b64_raw = res_dino['dino_vis_base64']
        
        dino_vis_b64_azure = dino_vis_b64_raw
        if not dino_vis_b64_azure.startswith("data:"):
            dino_vis_b64_azure = f"data:image/jpeg;base64,{dino_vis_b64_azure}"

        # 4. GPT Plan
        gpt_res2 = call_azure(PROMPT_STAGE_2, instruction, img_b64_raw)
        json_res2 = parse_json_gpt(gpt_res2)

        # 5. GPT Map
        targets = []
        for act in json_res2.get("action_components", []):
            comp = act.get("component", [])
            if len(comp) == 2 and str(comp[0]).lower() not in ['arm', 'robot']:
                targets.append(comp)
        
        prompt_user_3 = f"targets = {json.dumps(targets)}"
        gpt_res3 = call_azure(PROMPT_STAGE_3_TEMPLATE, prompt_user_3, dino_vis_b64_azure)
        json_res3 = parse_json_gpt(gpt_res3)

        # 6. Merge
        id_map = {}
        for item in json_res3.get("results", []):
            k = f"{str(item.get('object','')).strip().lower()}|{str(item.get('part','')).strip().lower()}"
            id_map[k] = item.get("id")
        
        coord_map = {kp['id']: [kp['x'], kp['y']] for kp in keypoints}

        final_actions = []
        for act in json_res2.get("action_components", []):
            new_act = act.copy()
            comp = act.get("component", [])
            
            if len(comp) == 2:
                obj_k = str(comp[0]).strip().lower()
                part_k = str(comp[1]).strip().lower()
                
                if obj_k in ['arm', 'robot']:
                    new_act['target_id'] = None
                    new_act['coords'] = None
                else:
                    k = f"{obj_k}|{part_k}"
                    tid = id_map.get(k)
                    if tid is not None:
                        new_act['target_id'] = tid
                        new_act['coords'] = coord_map.get(tid)
                    else:
                        new_act['target_id'] = None
                        new_act['coords'] = None
                        new_act['warning'] = "ID Not Found"
            final_actions.append(new_act)

        return {
            "status": "success",
            "instruction": instruction,
            "intent_analysis": json_res1,
            "raw_plan": json_res2,
            "id_mapping": json_res3,
            "final_action_plan": final_actions,
            "vis_images": {"sam": sam_vis_b64_raw, "dino": dino_vis_b64_raw}
        }

    except Exception as e:
        return {"status": "failed", "instruction": instruction, "error": str(e)}


# ============================================================
# 📂 主处理逻辑 (Granular Skip Existing)
# ============================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def process_single_npz(npz_path, output_dir, img_output_dir):
    """
    处理单个 NPZ 文件：
    1. 检查是否存在同名 JSON
    2. 加载 JSON，清洗掉之前失败的记录（以便重试）
    3. 仅保留成功的 task_index 跳过执行
    4. 增量更新保存
    """
    try:
        base_name = os.path.basename(npz_path).replace(".npz", "")
        json_name = f"{base_name}.json"
        save_path = os.path.join(output_dir, json_name)

        # 1. 读取 NPZ 数据
        data = np.load(npz_path, allow_pickle=True)
        if "image" not in data.files: return
        image_rgb = data["image"]
        
        trajectories = []
        if "trajectories" in data.files:
            raw_traj = data["trajectories"]
            s_traj = str(raw_traj.item()) if raw_traj.ndim == 0 else str(raw_traj)
            try: trajectories = json.loads(s_traj)
            except: trajectories = []

        if not trajectories: return

        # 2. 加载或初始化结果容器
        final_output = {}
        processed_indices = set()

        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    final_output = json.load(f)
                    
                    # =======================================================
                    # 🔥 修改核心：清洗历史数据
                    # 只保留成功的记录，丢弃失败的记录以便重试
                    # =======================================================
                    valid_results = []
                    existing_results = final_output.get("results", [])
                    
                    for res in existing_results:
                        status = res.get("ai_inference", {}).get("status")
                        task_idx = res.get("task_index")
                        
                        if status == "success":
                            # 成功的任务：保留并在本次运行中跳过
                            valid_results.append(res)
                            if task_idx is not None:
                                processed_indices.add(task_idx)
                        else:
                            # 失败的任务：不保留，不加入 processed_indices
                            # 这样稍后循环时会检测到它未完成，从而触发重试
                            err_msg = res.get("ai_inference", {}).get("error", "Unknown")
                            print(f"   ⚠️ Drop failed history Task {task_idx} (Error: {str(err_msg)[:30]}...). Preparing to retry.")
                    
                    # 将清洗后的列表赋值回 final_output，防止数据重复堆积
                    final_output["results"] = valid_results
                    # =======================================================

            except Exception as e:
                print(f"⚠️ JSON load error {json_name}, creating new. Error: {e}")
                final_output = {
                    "source_file": npz_path,
                    "image_filename": f"{base_name}.png",
                    "results": []
                }
        else:
            final_output = {
                "source_file": npz_path,
                "image_filename": f"{base_name}.png",
                "results": []
            }

        # 确保 results 列表存在
        if "results" not in final_output:
            final_output["results"] = []

        # 3. 遍历任务
        any_updates = False
        
        for idx, traj in enumerate(trajectories):
            # === 🟢 跳过逻辑：如果 ID 在已处理集合中，直接跳过 ===
            if idx in processed_indices:
                continue 
            
            instruction = traj.get("instruction", "").strip()
            if not instruction: continue
            
            print(f"   🔹 Running {base_name} [Task {idx}]: {instruction}")
            
            # === 执行推理 ===
            ai_result = run_inference_pipeline(image_rgb, instruction)
            
            # === 保存图片 ===
            if ai_result.get("status") == "success":
                vis_imgs = ai_result.get("vis_images", {})
                sam_fname = f"{base_name}_task{idx}_sam.jpg"
                dino_fname = f"{base_name}_task{idx}_dino.jpg"
                
                if "sam" in vis_imgs:
                    save_base64_image(vis_imgs["sam"], os.path.join(img_output_dir, sam_fname))
                if "dino" in vis_imgs:
                    save_base64_image(vis_imgs["dino"], os.path.join(img_output_dir, dino_fname))
                del ai_result["vis_images"]

            # 构造结果项
            task_record = {
                "task_index": idx,
                "original_instruction": instruction,
                "ai_inference": ai_result
            }
            
            # 添加到列表并标记更新
            final_output["results"].append(task_record)
            any_updates = True
            
            # === 🟢 增量保存 (每跑完一个任务就存一次) ===
            # 这里使用了 NpEncoder 确保 numpy 数据类型能被序列化
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, cls=NpEncoder, indent=2, ensure_ascii=False)
            
            # 简单延时防止请求过快
            time.sleep(0.5)

        if any_updates:
            print(f"   ✅ Updated: {save_path}")

    except Exception as e:
        print(f"❌ Error processing {npz_path}: {e}")

# === 并行包装 ===
def process_wrapper(args_tuple):
    npz_path, output_dir, img_output_dir = args_tuple
    try:
        process_single_npz(npz_path, output_dir, img_output_dir)
        return True
    except: return False

def main():
    parser = argparse.ArgumentParser(description="Batch process NPZ files (Granular Skip Existing)")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder (.npz)")
    parser.add_argument("--output_dir", type=str, default="ai_results_all", help="Output folder (.json)")
    parser.add_argument("--img_output_dir", type=str, default=None, help="Image output folder")
    parser.add_argument("--workers", type=int, default=2, help="Concurrency (Default: 2)")
    
    args = parser.parse_args()

    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise RuntimeError("Missing Azure config. Please set AZURE_ENDPOINT and AZURE_API_KEY.")
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    img_dir = args.img_output_dir if args.img_output_dir else os.path.join(args.output_dir, "vis_images")
    if not os.path.exists(img_dir): os.makedirs(img_dir)
        
    npz_files = glob.glob(os.path.join(args.input_dir, "*.npz"))
    
    print(f"🔍 Found {len(npz_files)} files.")
    print(f"   Skip Strategy: Granular (Check existing tasks inside JSON)")
    print(f"   Workers: {args.workers}")
    
    task_args = [(f, args.output_dir, img_dir) for f in npz_files]
    
    # ==========================================
    # 🔥 修改开始：增加 Ctrl+C 强制退出逻辑
    # ==========================================
    executor = None
    try:
        # 手动创建 executor，不使用 'with' 语句，以便完全控制关闭流程
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(process_wrapper, task) for task in task_args]
        
        # 监听进度
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass
            
        print("\n🎉 All done!")
        
        # 正常结束时，等待所有线程安全退出
        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        print("\n\n🛑 检测到 Ctrl+C！正在强制停止所有任务...")
        
        if executor:
            # 尝试取消未开始的任务 (Python 3.9+ 支持 cancel_futures)
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except:
                executor.shutdown(wait=False)
        
        # ☢️ 强制退出：直接杀掉进程，不等待子线程网络超时
        os._exit(0)
    # ==========================================
    # 🔥 修改结束
    # ==========================================

if __name__ == "__main__":
    main()
