import os
import sys
import cv2
import json
import mujoco
import matplotlib.pyplot as plt 
import time
import spatialmath as sm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

from vlm_process import  segment_and_crop_mask,get_circular_area_mask,choose_model, process_sam_results, generate_robot_actions, _bbox_center
from grasp_process import run_grasp_inference, GraspFlow

# Planner switch via env var (method-oriented naming):
# - VLM_METHOD=svp   -> use vision_agent_v2 (default, paper main method path)
# - VLM_METHOD=qwen  -> use vision_agent
# Backward compatibility: if VLM_METHOD is not set, fallback to VLM_AGENT_IMPL.
_method_raw = os.getenv("VLM_METHOD") or os.getenv("VLM_AGENT_IMPL", "svp")
_agent_impl = _method_raw.strip().lower()

if _agent_impl in {"qwen", "vision_agent", "v1"}:
    from vision_agent import plan_and_ground_robot_actions
    _selected_agent = "vision_agent"
    _selected_method = "qwen"
elif _agent_impl in {"svp", "vision_agent_v2", "v2", "azure"}:
    from vision_agent_v2 import plan_and_ground_robot_actions
    _selected_agent = "vision_agent_v2"
    _selected_method = "svp"
else:
    raise ValueError(
        "Invalid method. Use VLM_METHOD=svp or VLM_METHOD=qwen "
        "(compatible aliases: vision_agent_v2, v2, azure, vision_agent, v1)."
    )

# 全局变量
global color_img, depth_img, env
color_img = None
depth_img = None
env = None


#获取彩色和深度图像数据
def get_image(env):
    global color_img, depth_img
     # 从环境渲染获取图像数据
    imgs = env.render()

    # 提取彩色和深度图像数据
    color_img = imgs['img']   # 这是RGB格式的图像数据
    depth_img = imgs['depth'] # 这是深度数据

    # 将RGB图像转换为OpenCV常用的BGR格式
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    return color_img, depth_img

#构造回调函数，不断调用
def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()

import json
import time
import numpy as np
import spatialmath as sm

# def execute_actions_from_res(res, color_img, depth_img, env):
#     """
#     根据 res["action_json_str"] 依次将动作派发到 flow
    
#     ★ 关键改进：在任何机械臂运动前，预先计算所有像素坐标
#     """
#     if color_img is None or depth_img is None:
#         print("[WARNING] Waiting for image data...")
#         return

#     # 步骤1: 解析动作序列
#     action_data = res.get("action_json_str", None)
#     if action_data is None:
#         raise ValueError("res['action_json_str'] 不存在")

#     if isinstance(action_data, str):
#         action_data = json.loads(action_data)

#     if isinstance(action_data, dict) and "action_components" in action_data:
#         actions = action_data["action_components"]
#     elif isinstance(action_data, list):
#         actions = action_data
#     else:
#         raise ValueError("action_json_str 的结构不正确")

#     def _fmt(obj):
#         try:
#             return json.dumps(obj, ensure_ascii=False)
#         except Exception:
#             return str(obj)

#     print("\n========== 执行动作清单 ==========")
#     for idx, item in enumerate(actions):
#         print(f"[计划] #{idx}: action={item.get('action')}  "
#               f"component={_fmt(item.get('component'))}  "
#               f"description={item.get('description')}")
#     print("================================\n")

#     #  步骤2: 预计算所有像素坐标 
#     print("\n" + "="*60)
#     print("📸 预计算阶段：在机械臂移动前计算所有目标坐标")
#     print("="*60)
    
#     # 2.1 提取所有需要坐标的像素点
#     pixel_list = []
#     action_pixel_map = {}  # {动作索引: 像素坐标}
    
#     for idx, item in enumerate(actions):
#         args = item.get("args", {}) or {}
#         pixel = args.get("pixel", None)
        
#         if pixel is not None:
#             if not (isinstance(pixel, (list, tuple)) and len(pixel) == 2):
#                 print(f"[WARN] 动作 #{idx} 的 pixel 格式不正确: {pixel}")
#                 continue
            
#             u, v = int(pixel[0]), int(pixel[1])
#             pixel_tuple = (u, v)
            
#             # 去重记录
#             if pixel_tuple not in pixel_list:
#                 pixel_list.append(pixel_tuple)
            
#             action_pixel_map[idx] = pixel_tuple
    
#     print(f"[INFO] 检测到 {len(pixel_list)} 个需要坐标转换的像素点: {pixel_list}")
    
#     # 2.2 计算相机到基座的变换矩阵
#     n_wc = np.array([0.0, -1.0, 0.0])
#     o_wc = np.array([-1.0, 0.0, -0.5])
#     t_wc = np.array([0.85, 0.8, 1.6])
#     T_base_cam = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    
#     # 2.3 计算内参
#     h, w = depth_img.shape[:2]
#     fovy = np.pi / 4
#     focal = h / (2.0 * np.tan(fovy / 2.0))
#     K = (focal, focal, w/2.0, h/2.0)
    
#     print(f"[INFO] 图像尺寸: {w}x{h}")
#     print(f"[INFO] 相机内参: fx={K[0]:.2f}, fy={K[1]:.2f}, cx={K[2]:.2f}, cy={K[3]:.2f}")
    
#     # 2.4 批量计算坐标（此时机械臂还未移动，场景无遮挡）
#     coord_cache = {}
    
#     for u, v in pixel_list:
#         # 检查边界
#         if not (0 <= u < w and 0 <= v < h):
#             print(f"[WARN] 像素 ({u},{v}) 超出图像范围，跳过")
#             continue
        
#         # 直接使用中心点深度
#         Zm = float(depth_img[v, u])
        
#         if Zm <= 0 or not np.isfinite(Zm):
#             print(f"[WARN] 像素 ({u},{v}) 深度无效: {Zm}")
#             continue
        
#         # 像素→基座坐标转换
#         xb, yb, zb = _pixel2base_xyz(u, v, Zm, K, T_base_cam)
#         coord_cache[(u, v)] = (xb, yb, zb)
        
#         print(f"  [✓] 像素 ({u:3d},{v:3d}) → 深度 {Zm:.4f}m → "
#               f"基座 ({xb:.4f}, {yb:.4f}, {zb:.4f})")
    
#     print(f"[INFO] 成功预计算 {len(coord_cache)} 个坐标")
#     print("="*60 + "\n")
    
#     if len(coord_cache) == 0 and len(pixel_list) > 0:
#         raise RuntimeError("所有像素点的坐标转换都失败了！")

#     # 步骤3: 执行动作序列（使用预计算的坐标）
#     flow = None
#     gg = None
#     masks = None

#     for i, item in enumerate(actions):
#         action = item.get("action")
#         args = item.get("args", {}) or {}
#         pixel = args.get("pixel", None)

#         # 统一把 pixel 转成 tuple
#         if pixel is not None:
#             if not (isinstance(pixel, (list, tuple)) and len(pixel) == 2):
#                 raise ValueError(f"第 {i} 个动作的 pixel 格式不正确: {pixel}")
#             point_xy = (int(pixel[0]), int(pixel[1]))
#         else:
#             point_xy = None

#         t0 = time.time()
#         print(f"\n--- 执行动作 #{i} | {action} ---")
#         if point_xy is not None:
#             print(f"[输入参数] pixel={point_xy}")
#         if args and (pixel is None or len(args) > 1):
#             rest_args = {k: v for k, v in args.items() if k != "pixel"}
#             if rest_args:
#                 print(f"[输入参数] 其他 args={_fmt(rest_args)}")

#         # ========== move_to_grasp_pose ==========
#         if action == "move_to_grasp_pose":
#             if point_xy is None:
#                 raise ValueError(f"第 {i} 个动作 move_to_grasp_pose 缺少 pixel")

#             save_path = "mask_final.png"
#             print(f"[步骤] 生成目标 mask -> 保存={save_path}")
#             # masks = segment_and_crop_mask(
#             #     image_bgr=color_img,
#             #     point_xy=point_xy,
#             #     radius_px=30,
#             #     save_path=save_path
#             # )
#             masks = get_circular_area_mask(
#                 image_bgr=color_img,
#                 point_xy=point_xy,
#                 radius_px=30, # 这里的半径决定了给 GraspNet 的数据范围
#                 depth=depth_img # 建议传入深度图以过滤背景噪音
#             )
#             print("[步骤] 运行抓取推理 run_grasp_inference(...)")
#             gg = run_grasp_inference(color_img, depth_img, masks)

#             try:
#                 print(f"[信息] 抓取候选数量: {len(gg)}")
#             except Exception:
#                 print("[信息] 抓取候选数量: <未知类型，无法统计>")

#             if flow is None:
#                 flow = GraspFlow(env, gg)
#                 print("[状态] 初始化 GraspFlow 完成")

#             flow.move_to_grasp_pose()
#             print("[结果] move_to_grasp_pose 完成")

#         # ========== move_to_point ==========
#         elif action == "move_to_point":
#             if point_xy is None:
#                 raise ValueError(f"第 {i} 个动作 move_to_point 缺少 pixel")
#                 # 若还没有 flow（例如序列一上来就是 move_to_point），这里临时初始化
#             if flow is None:
#                 flow = GraspFlow.from_env(env)
#                 print("[状态] 初始化 GraspFlow（无抓取）完成")
            
#             # 使用预计算的基座坐标（避免遮挡）
#             if point_xy in coord_cache:
#                 target_base_coord = coord_cache[point_xy]
#                 print(f"[步骤] 使用预计算的基座坐标: {target_base_coord}")
#                 print(f"       (原像素坐标: {point_xy})")
                
#                 # 传入基座坐标（3元组），而不是像素坐标
#                 flow.move_to_point(target_base_coord)
#                 print("[结果] move_to_point 完成（已到达目标位置）")
#             else:
#                 print(f"[ERROR] 像素 {point_xy} 的坐标未在预计算中找到！")
#                 raise RuntimeError(f"缺少坐标缓存: {point_xy}")

#         # ========== close_gripper ==========
#         elif action == "close_gripper":
#             print("[步骤] 关闭夹爪")
#             flow.close_gripper()
#             print("[结果] close_gripper 完成")

#         # ========== open_gripper ==========
#         elif action == "open_gripper":
#             print("[步骤] 打开夹爪")
#             flow.open_gripper()
#             print("[结果] open_gripper 完成")

#         # ========== back_to_zero ==========
#         elif action == "back_to_zero":
#             print("[步骤] 回到初始位姿")
#             flow.back_to_zero()
#             print("[结果] back_to_zero 完成")

#         else:
#             print(f"[WARN] 未知动作 '{action}'，跳过。")

#         dt = time.time() - t0
#         print(f"[耗时] {dt:.3f}s  —— #{i} {action}")

#     print("\n=== 全部动作执行完成 ===\n")

def execute_actions_from_res(res, color_img, depth_img, env):
    """
    根据 res["action_json_str"] 依次将动作派发到 flow
    
    ★ 融合策略：
      1. 预计算：移动前计算点击点的 3D 坐标（避免遮挡）。
      2. 抓取策略：GraspNet 获取大范围（80px）点云上下文，但结果被约束在小范围（30px）点击点附近。
    """
    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # =========================================================================
    # 步骤1: 解析动作序列
    # =========================================================================
    action_data = res.get("action_json_str", None)
    if action_data is None:
        raise ValueError("res['action_json_str'] 不存在")

    if isinstance(action_data, str):
        action_data = json.loads(action_data)

    if isinstance(action_data, dict) and "action_components" in action_data:
        actions = action_data["action_components"]
    elif isinstance(action_data, list):
        actions = action_data
    else:
        raise ValueError("action_json_str 的结构不正确")

    def _fmt(obj):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    print("\n========== 执行动作清单 ==========")
    for idx, item in enumerate(actions):
        print(f"[计划] #{idx}: action={item.get('action')}  "
              f"component={_fmt(item.get('component'))}  "
              f"description={item.get('description')}")
    print("================================\n")

    # =========================================================================
    # 步骤2: 预计算所有像素坐标 (在机械臂移动前完成)
    # =========================================================================
    print("\n" + "="*60)
    print("📸 预计算阶段：在机械臂移动前计算所有目标坐标")
    print("="*60)
    
    # 2.1 提取所有需要坐标的像素点
    pixel_list = []
    
    for idx, item in enumerate(actions):
        args = item.get("args", {}) or {}
        pixel = args.get("pixel", None)
        
        if pixel is not None:
            if not (isinstance(pixel, (list, tuple)) and len(pixel) == 2):
                print(f"[WARN] 动作 #{idx} 的 pixel 格式不正确: {pixel}")
                continue
            
            u, v = int(pixel[0]), int(pixel[1])
            pixel_tuple = (u, v)
            
            # 去重记录
            if pixel_tuple not in pixel_list:
                pixel_list.append(pixel_tuple)
    
    print(f"[INFO] 检测到 {len(pixel_list)} 个需要坐标转换的像素点: {pixel_list}")
    
    # 2.2 计算相机到基座的变换矩阵 (请根据实际标定修改)
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([0.85, 0.8, 1.6])
    T_base_cam = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    
    # 2.3 计算内参
    h, w = depth_img.shape[:2]
    fovy = np.pi / 4
    focal = h / (2.0 * np.tan(fovy / 2.0))
    K = (focal, focal, w/2.0, h/2.0)
    
    print(f"[INFO] 图像尺寸: {w}x{h}")
    
    # 2.4 批量计算坐标（此时机械臂还未移动，场景无遮挡）
    coord_cache = {}
    
    for u, v in pixel_list:
        # 检查边界
        if not (0 <= u < w and 0 <= v < h):
            print(f"[WARN] 像素 ({u},{v}) 超出图像范围，跳过")
            continue
        
        # 直接使用中心点深度
        Zm = float(depth_img[v, u])
        
        if Zm <= 0 or not np.isfinite(Zm):
            print(f"[WARN] 像素 ({u},{v}) 深度无效: {Zm}")
            continue
        
        # 像素→基座坐标转换
        xb, yb, zb = _pixel2base_xyz(u, v, Zm, K, T_base_cam)
        coord_cache[(u, v)] = (xb, yb, zb)
        
        print(f"  [✓] 像素 ({u:3d},{v:3d}) → 深度 {Zm:.4f}m → "
              f"基座 ({xb:.4f}, {yb:.4f}, {zb:.4f})")
    
    print(f"[INFO] 成功预计算 {len(coord_cache)} 个坐标")
    print("="*60 + "\n")
    
    # 若有需要移动到点但没算出坐标，提前报错比撞车好
    # if len(coord_cache) == 0 and len(pixel_list) > 0:
    #     print("[WARN] 所有像素点的坐标转换都失败了，后续 move_to_point 可能失败！")

    # =========================================================================
    # 步骤3: 执行动作序列 (应用新策略)
    # =========================================================================
    flow = None
    gg = None
    
    # 定义策略参数：大视野感知，小范围约束
    CONTEXT_RADIUS_PX = 80    # 生成 Mask 给 GraspNet 看的范围（大）
    CONSTRAINT_RADIUS_PX = 30 # 最终筛选抓取点的范围（小）

    for i, item in enumerate(actions):
        action = item.get("action")
        args = item.get("args", {}) or {}
        pixel = args.get("pixel", None)

        # 统一解析 pixel
        if pixel is not None:
            point_xy = (int(pixel[0]), int(pixel[1]))
        else:
            point_xy = None

        t0 = time.time()
        print(f"\n--- 执行动作 #{i} | {action} ---")
        if point_xy is not None:
            print(f"[输入参数] pixel={point_xy}")

        # ---------------------------------------------------------------------
        # Action: move_to_grasp_pose
        # ---------------------------------------------------------------------
        if action == "move_to_grasp_pose":
            if point_xy is None:
                raise ValueError(f"第 {i} 个动作 move_to_grasp_pose 缺少 pixel")

            # 1. 生成【上下文掩码】(Wide Context)
            print(f"[步骤] 生成上下文掩码 (Radius: {CONTEXT_RADIUS_PX})")
            context_mask = get_circular_area_mask(
                image_bgr=color_img,
                point_xy=point_xy,
                radius_px=CONTEXT_RADIUS_PX, # 宽输入：为了看清物体结构
                depth=depth_img 
            )

            # 2. 运行推理并应用【区域约束】(Narrow Constraint)
            print(f"[步骤] 运行抓取推理 (约束半径: {CONSTRAINT_RADIUS_PX})")
            # 注意：这里假设你的 run_grasp_inference 已经更新为支持 constraint_radius 参数
            gg = run_grasp_inference(
                color_path=color_img,      # 如果接口是传图片对象，这里传对象
                depth_path=depth_img, 
                sam_mask_path=context_mask,   # 传入大掩码
                target_point=point_xy,        # 传入目标中心
                constraint_radius=CONSTRAINT_RADIUS_PX # 传入约束半径
            )

            try:
                print(f"[信息] 筛选后有效抓取数量: {len(gg)}")
            except Exception:
                print("[信息] 抓取结果无效或为空")

            if gg is None or len(gg) == 0:
                print("[ERROR] 未找到符合约束的抓取姿态！跳过此动作。")
                continue

            if flow is None:
                flow = GraspFlow(env, gg)
                print("[状态] 初始化 GraspFlow 完成")
            else:
                # 如果 flow 已存在（比如之前的动作初始化过），更新它的 gg
                flow.gg = gg

            flow.move_to_grasp_pose()
            print("[结果] move_to_grasp_pose 完成")

        # ---------------------------------------------------------------------
        # Action: move_to_point
        # ---------------------------------------------------------------------
        elif action == "move_to_point":
            if point_xy is None:
                raise ValueError(f"第 {i} 个动作 move_to_point 缺少 pixel")
                
            if flow is None:
                flow = GraspFlow.from_env(env)
                print("[状态] 初始化 GraspFlow（无抓取）完成")
            
            # 优先使用步骤2中预计算的基座坐标（无遮挡）
            if point_xy in coord_cache:
                target_base_coord = coord_cache[point_xy]
                print(f"[步骤] 使用预计算基座坐标: {target_base_coord}")
                print(f"       (原像素: {point_xy})")
                
                flow.move_to_point(target_base_coord)
                print("[结果] move_to_point 完成（已到达目标位置）")
            else:
                print(f"[ERROR] 像素 {point_xy} 的坐标未在预计算缓存中！")
                print("[WARN] 尝试使用 flow 内部转换（可能因遮挡而不准）...")
                # 这种情况下通常是因为 pixel 点深度无效或越界
                flow.move_to_point(point_xy) # 这是一个备选方案，如果有的话

        # ---------------------------------------------------------------------
        # Action: Gripper Controls & Reset
        # ---------------------------------------------------------------------
        elif action == "close_gripper":
            print("[步骤] 关闭夹爪")
            if flow: flow.close_gripper()
            print("[结果] close_gripper 完成")

        elif action == "open_gripper":
            print("[步骤] 打开夹爪")
            if flow: flow.open_gripper()
            print("[结果] open_gripper 完成")

        elif action == "back_to_zero":
            print("[步骤] 回到初始位姿")
            if flow: flow.back_to_zero()
            print("[结果] back_to_zero 完成")

        else:
            print(f"[WARN] 未知动作 '{action}'，跳过。")

        dt = time.time() - t0
        print(f"[耗时] {dt:.3f}s  —— #{i} {action}")

    print("\n=== 全部动作执行完成 ===\n")
    
# 辅助函数：像素→基座坐标转换（需要在文件开头定义）
def _pixel2base_xyz(u, v, Z_m, K, T_base_cam):
    """
    (u,v,Z_m) → 基座系(x,y,z)，单位米
    K = (fx, fy, cx, cy)
    T_base_cam: spatialmath.SE3 或 4x4/3x3 ndarray
    """
    fx, fy, cx, cy = K
    Xc = (u - cx) / fx * Z_m
    Yc = (v - cy) / fy * Z_m
    Zc = Z_m
    p_cam = np.array([Xc, Yc, Zc], dtype=np.float32)

    # 1) SE3 对象：用 R @ p + t
    if isinstance(T_base_cam, sm.SE3):
        p_base = T_base_cam.R @ p_cam + T_base_cam.t
        return float(p_base[0]), float(p_base[1]), float(p_base[2])

    # 2) ndarray：支持 4x4 或 3x3
    T = np.asarray(T_base_cam)
    if T.shape == (4, 4):
        p_h = T @ np.array([Xc, Yc, Zc, 1.0], dtype=np.float32)
        return float(p_h[0]), float(p_h[1]), float(p_h[2])
    elif T.shape == (3, 3):
        p_base = T @ p_cam
        return float(p_base[0]), float(p_base[1]), float(p_base[2])

    raise ValueError(f"T_base_cam 形状不支持: {getattr(T_base_cam, 'shape', type(T_base_cam))}")

def test_grasp():
    global color_img, depth_img, env

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # 用户输入指令 
    instruction = input("请输入操作指令（例如：将小黄鸭放在魔方上面）: ").strip()
    if not instruction:
        print("[提示] 未输入指令，默认执行：将小黄鸭放在魔方上面")
        instruction = "将小黄鸭放在魔方上面"

    # 调用多模态规划
    res = plan_and_ground_robot_actions(
        instruction=instruction,
        image_path=color_img,              # ndarray / PIL / path 都行
        save_vis_prefix="logs/run1",
        log_path="logs/run1.jsonl",        # ✅ 新增：写日志
        log_max_text_len=4000,             # 可选
        log_include_base64=False           # 可选（建议 False）
    )

    print(f"【用户指令】{instruction}")
    print("【规划输出】", res["action_json_str"])
    print("【可视化】", res["vis_path"])
    print("【日志】logs/run1.jsonl")

    # 执行动作序列
    execute_actions_from_res(res, color_img, depth_img, env)
    # execute_actions_from_res(res, color_img, depth_img, env, grasp_mode="vertical")


if __name__ == '__main__':
    
    env = UR5GraspEnv()
    env.reset()
    print(
        f"[INFO] planner method = {_selected_method}, "
        f"agent = {_selected_agent} (raw={_method_raw})"
    )
    
    while True:

        for i in range(500): # 1000
            env.step()

        color_img, depth_img = get_image(env)

        callback(color_img, depth_img)


    env.close()


    
