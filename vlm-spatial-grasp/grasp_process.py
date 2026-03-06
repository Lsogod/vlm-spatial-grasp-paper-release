import os
import sys
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import spatialmath as sm
import time

from manipulator_grasp.arm.motion_planning import *

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


#网络加载
def get_net(): 
    """
    加载训练好的 GraspNet 模型
    """
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.05, 
                   hmin=-0.02, 
                   hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


#  数据处理并生成输入 
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path[..., ::-1].astype(np.float32) / 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    height = color.shape[0]
    width = color.shape[1]
    fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask = depth < 2.0
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 5000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    # end_points = {'point_clouds': cloud_sampled}

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d


# 主函数：获取抓取预测 
def filter_grasps_by_pixel_area(gg, intrinsic, target_uv, radius_px):
    """
    过滤抓取：将3D抓取点投影回2D图像，只保留落在 target_uv + radius_px 范围内的抓取。
    
    参数:
    - gg: GraspGroup 对象
    - intrinsic: 相机内参矩阵 (3x3)
    - target_uv: 用户点击的中心点 (u, v) 像素坐标
    - radius_px: 允许抓取的像素半径 (约束范围)
    """
    if len(gg) == 0:
        return gg
        
    # 获取相机内参 fx, fy, cx, cy
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    filtered_indices = []
    
    # 遍历所有抓取
    for i, grasp in enumerate(gg):
        # 获取抓取中心点 (Camera Frame: x, y, z)
        x, y, z = grasp.translation
        
        # 排除无效深度
        if z <= 0: continue
            
        # 3D -> 2D 投影
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        
        # 计算与目标中心点 (target_uv) 的像素距离
        dist = np.sqrt((u - target_uv[0])**2 + (v - target_uv[1])**2)
        
        # 判断是否在约束范围内
        if dist <= radius_px:
            filtered_indices.append(i)
            
    # 返回筛选后的 GraspGroup
    print(f"[Filter] 原始抓取: {len(gg)} -> 区域约束后: {len(filtered_indices)} (Radius: {radius_px}px)")
    return gg[filtered_indices]

# 修改后的函数签名，增加了 target_point 和 constraint_radius
def run_grasp_inference(color_path, depth_path, sam_mask_path=None, 
                        target_point=None, constraint_radius=30):
    """
    target_point: (u, v) 像素坐标，用户点击的位置
    constraint_radius: 像素半径，只保留这个半径内的抓取
    """
    # 1. 加载网络
    net = get_net()

    # 2. 处理数据 (注意：这里的 sam_mask_path 应该是一个较大的 Context Mask，比如 60px)
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, sam_mask_path)

    # ... (原有内参计算代码需保留，或者从 get_and_process_data 返回 intrinsic) ...
    # 为了过滤，我们需要这里重新获取或计算 intrinsic，这里简单复刻一下计算逻辑
    if isinstance(color_path, str):
        img_h, img_w = np.array(Image.open(color_path)).shape[:2]
    else:
        img_h, img_w = color_path.shape[:2]
    
    fovy = np.pi / 4
    focal = img_h / (2.0 * np.tan(fovy / 2.0))
    intrinsic = np.array([[focal, 0, img_w/2], [0, focal, img_h/2], [0, 0, 1]])

    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 4. 构造 GraspGroup
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 5. 碰撞检测 (剔除与大掩码范围内点云碰撞的抓取)
    COLLISION_THRESH = 0.01
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
        gg = gg[~collision_mask]

    # 6. NMS 去重
    gg = gg.nms()

    # 7. 区域约束过滤 (Pixel Filter)
    if target_point is not None:
        gg = filter_grasps_by_pixel_area(gg, intrinsic, target_point, constraint_radius)
        if len(gg) == 0:
             print("[Warning] 区域约束后无结果，无法抓取。")
             return None

    # 8. 按照得分排序
    gg = gg.sort_by_score()

    # ===== 垂直角度过滤 =====
    # 将 gg 转换为列表进行遍历
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1]) 
    angle_threshold = np.deg2rad(30)
    filtered = []
    
    for grasp in all_grasps:
        approach_dir = grasp.rotation_matrix[:, 0]
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    
    if len(filtered) > 0:
        # === 🔴 修复点：正确地从 Grasp 对象列表中提取数据重建 GraspGroup ===
        # 错误写法: gg = GraspGroup(np.array(filtered)) 
        # 正确写法: 提取每个 grasp 对象的 .grasp_array (它是 (17,) 的 numpy 数组)
        new_grasp_array = np.array([g.grasp_array for g in filtered])
        gg = GraspGroup(new_grasp_array)
        print(f"[DEBUG] 角度过滤后保留: {len(gg)} 个抓取")
    else:
        print("[Warning] 垂直角度过滤后无结果，保留原结果")
        # 这种情况下 gg 保持原样，不用变

    # ===== 最终取最佳抓取 =====
    if len(gg) > 0:
        best_grasp = gg[0] # 现在这里不会报错了
    else:
        print("未找到有效抓取")
        return None

    # ===== 既然已经用了 target_point 过滤，后面的 object_center 距离排序可以简化 =====
    # 因为所有剩下的抓取都已经离 target_point 很近了 (<= 30px)
    # 我们直接取 Score 最高的即可
    
    if len(gg) > 0:
        best_grasp = gg[0] # 取分数最高的
    else:
        print("未找到有效抓取")
        return None

    # 创建返回结果
    new_gg = GraspGroup()
    new_gg.add(best_grasp)

    # 可视化
    visual = True
    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    return new_gg

# 对graspnet生产的抓取结果进行过滤和排序，返回最终的GraspGroup对象
# def run_grasp_inference(color_path, depth_path, sam_mask_path=None):
#     # 1. 加载网络
#     net = get_net()

#     # 2. 处理数据，此处使用返回的工作空间掩码路径
#     end_points, cloud_o3d = get_and_process_data(color_path, depth_path, sam_mask_path)

#     # 3. 前向推理
#     with torch.no_grad():
#         end_points = net(end_points)
#         grasp_preds = pred_decode(end_points)

#     # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
#     gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

#     # 5. 碰撞检测
#     COLLISION_THRESH = 0.01
#     if COLLISION_THRESH > 0:
#         voxel_size = 0.01
#         collision_thresh = 0.01
#         mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
#         collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
#         gg = gg[~collision_mask]

#     # 6. NMS 去重 + 按照得分排序（降序）
#     gg.nms().sort_by_score()

#     # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
#     # 将 gg 转换为普通列表
#     all_grasps = list(gg)
#     vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
#     angle_threshold = np.deg2rad(30)  # 30度的弧度值
#     filtered = []
#     for grasp in all_grasps:
#         # 抓取的接近方向取 grasp.rotation_matrix 的第一列
#         approach_dir = grasp.rotation_matrix[:, 0]
#         # 计算夹角：cos(angle)=dot(approach_dir, vertical)
#         cos_angle = np.dot(approach_dir, vertical)
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         angle = np.arccos(cos_angle)
#         if angle < angle_threshold:
#             filtered.append(grasp)
#     if len(filtered) == 0:
#         print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
#         filtered = all_grasps
#     else:
#         print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

#     # # ===== 新增：利用 SAM 生成的目标掩码过滤抓取预测（投影到图像坐标判断） =====
#     # if sam_mask_path is not None:
#     #     # 加载 SAM 目标掩码
#     #     if isinstance(sam_mask_path, str):
#     #         sam_mask = np.array(Image.open(sam_mask_path))
#     #     elif isinstance(sam_mask_path, np.ndarray):
#     #         sam_mask = sam_mask_path
#     #     else:
#     #         raise TypeError("sam_mask_path 既不是字符串路径也不是 NumPy 数组！")
#     #     # 假定 SAM 掩码与颜色图尺寸一致（640x640）
#     #     height, width = sam_mask.shape[:2]
#     #     # 动态计算相机内参（基于垂直视场角 fovy）
#     #     fovy = np.pi / 4  # 垂直视场角，假设为 45 度
#     #     focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（像素单位）
#     #     cx = width / 2.0   # 光心 X 坐标（图像中心）
#     #     cy = height / 2.0  # 光心 Y 坐标（图像中心）

#     #     sam_filtered = []
#     #     for grasp in filtered:
#     #         # grasp.translation 为摄像头坐标系下的 3D 坐标 [X, Y, Z]
#     #         X, Y, Z = grasp.translation
#     #         if Z <= 0:
#     #             continue
#     #         u = focal * X / Z + cx
#     #         v = focal * Y / Z + cy
#     #         u_int = int(round(u))
#     #         v_int = int(round(v))
#     #         # 检查投影点是否在图像范围内（640x640）
#     #         if u_int < 0 or u_int >= 640 or v_int < 0 or v_int >= 640:
#     #             continue
#     #         # 若 SAM 掩码中该像素有效（非0），则保留
#     #         if sam_mask[v_int, u_int] > 0:
#     #             sam_filtered.append(grasp)
#     #     if len(sam_filtered) == 0:
#     #         print("\n[Warning] No grasp predictions fall inside the SAM mask. Using previous predictions.")
#     #     else:
#     #         print(f"\n[DEBUG] Filtered {len(sam_filtered)} grasps inside the SAM mask out of {len(filtered)} predictions.")
#     #         filtered = sam_filtered

#     # ===== 新增部分：计算物体中心点 =====
#     # 使用点云计算物体的中心点
#     points = np.asarray(cloud_o3d.points)
#     object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

#     # 计算每个抓取位姿中心点与物体中心点的距离
#     distances = []
#     for grasp in filtered:
#         grasp_center = grasp.translation
#         distance = np.linalg.norm(grasp_center - object_center)
#         distances.append(distance)

#     # 创建一个新的排序列表，包含距离和抓取对象
#     grasp_with_distances = [(g, d) for g, d in zip(filtered, distances)]
    
#     # 按距离升序排序（距离越小越好）
#     grasp_with_distances.sort(key=lambda x: x[1])
    
#     # 提取排序后的抓取列表
#     filtered = [g for g, d in grasp_with_distances]

#     # ===== 新增部分：综合得分和距离进行最终排序 =====
#     # 创建一个新的排序列表，包含综合得分和抓取对象
#     # 综合得分 = 抓取得分 * 0.7 + (1 - 距离/最大距离) * 0.3
#     max_distance = max(distances) if distances else 1.0
#     grasp_with_composite_scores = []

#     for g, d in grasp_with_distances:
#         # 归一化距离分数（距离越小分数越高）
#         distance_score = 1 - (d / max_distance)
        
#         # 综合得分 = 抓取得分 * 权重1 + 距离得分 * 权重2
#         composite_score = g.score * 0.1 + distance_score * 0.9
#         # print(f"\n g.score = {g.score}, distance_score = {distance_score}")
#         grasp_with_composite_scores.append((g, composite_score))

#     # 按综合得分降序排序
#     grasp_with_composite_scores.sort(key=lambda x: x[1], reverse=True)

#     # 提取排序后的抓取列表
#     filtered = [g for g, score in grasp_with_composite_scores]


#     # # 对过滤后的抓取根据 score 排序（降序）
#     # filtered.sort(key=lambda g: g.score, reverse=True)

#     # 取第1个抓取
#     top_grasps = filtered[:1]

#     # 可视化过滤后的抓取，手动转换为 Open3D 物体
#     grippers = [g.to_open3d_geometry() for g in top_grasps]

#     # 选择得分最高的抓取（filtered 列表已按得分降序排序）
#     best_grasp = top_grasps[0]
#     best_translation = best_grasp.translation
#     best_rotation = best_grasp.rotation_matrix
#     best_width = best_grasp.width

#     # 创建一个新的 GraspGroup 并添加最佳抓取
#     new_gg = GraspGroup()            # 初始化空的 GraspGroup
#     new_gg.add(best_grasp)           # 添加最佳抓取

#     visual = True
#     if visual:
#         grippers = new_gg.to_open3d_geometry_list()
#         o3d.visualization.draw_geometries([cloud_o3d, *grippers])

#     return new_gg

#     #return best_translation, best_rotation, best_width

# 工具：通用执行器
def _run_planners(robot, env, planner_array, time_array, action, dt=0.002):
    """
    与原代码完全一致的执行逻辑：
    - 不创建新的 action，沿用并更新传入的 action（保证与原来“先前一步的关节位姿”连续）。
    - 关节/笛卡尔两用。
    """
    total_time = np.sum(time_array)
    time_step_num = round(total_time / dt) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)

    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    return action  # 继续复用

def _depth_at_median(depth_img, u, v, win=3, depth_scale=1.0):
    import numpy as np
    h, w = depth_img.shape[:2]
    u0, v0 = int(round(u)), int(round(v))
    u1, u2 = max(0, u0 - win), min(w, u0 + win + 1)
    v1, v2 = max(0, v0 - win), min(h, v0 + win + 1)
    patch = depth_img[v1:v2, u1:u2].astype(np.float32)
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        raise ValueError(f"像素({u},{v})邻域无有效深度")
    return float(np.median(vals)) * float(depth_scale)

def _pixel2base_xyz(u, v, Z_m, K, T_base_cam):
    """
    (u,v,Z_m) → 基座系(x,y,z)，单位米
    K = (fx, fy, cx, cy)
    T_base_cam: 可以是 spatialmath.SE3 或 4x4/3x3 ndarray
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

# 基础步骤封装
def step1_prepose(robot, env, q0, q1, time1, action, dt=0.002):
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    time_array = [0.0, time1]
    planner_array = [planner1]
    return _run_planners(robot, env, planner_array, time_array, action, dt)


def step2_approach(robot, env, T1, T2, time2, action, dt=0.002):
    position_parameter1 = LinePositionParameter(T1.t, T2.t)
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time2)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    planner2 = TrajectoryPlanner(trajectory_parameter1)
    time_array = [0.0, time2]
    planner_array = [planner2]
    action = _run_planners(robot, env, planner_array, time_array, action, dt)
    return action


def step3_to_grasp(robot, env, T2, T3, time3, action, dt=0.002):
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    time_array = [0.0, time3]
    planner_array = [planner3]
    return _run_planners(robot, env, planner_array, time_array, action, dt)

def move_to_grasp_pose(robot, env,
                       q0, q1, T_wo,
                       time1=1.0, time2=1.0, time3=1.0,
                       action=None, dt=0.002,
                       approach_offset=(-0.1, 0.0, 0.0)):
    if action is None:
        action = np.zeros(7)
    # 1) 预抓取
    action = step1_prepose(robot, env, q0, q1, time1, action, dt)
    # 2) 接近（保持原来 robot.set_joint(q1) 的行为）
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(*approach_offset)
    action = step2_approach(robot, env, T1, T2, time2, action, dt)
    # 3) 到抓取位姿
    T3 = T_wo
    action = step3_to_grasp(robot, env, T2, T3, time3, action, dt)
    return action, T2, T3


def close_gripper(env, action, loops=1000, delta=0.2):
    """
    与原逻辑完全一致：仅改 action[-1]，持续复用上一步留下的 action[:6]
    """
    for _ in range(loops):
        action[-1] += delta
        action[-1] = np.min([action[-1], 255])
        env.step(action)
    return action


def move_to_point(robot, env,
                  T_start,               # 抓起后的起点位姿（一般是 T3/T3_meas）
                  target,                # 目标点：支持 (x,y)、(x,y,z) 或 SE3
                  time4=1.0, time5=1.0, time6=1.0,
                  action=None, dt=0.002,
                  lift_height=0.3,       # 第4步：向上抬高
                  descend_dist=0.25,      # 第6步：向下放置距离
                  keep_orientation=True  # 是否保持 T_start 的朝向
                  ):
    """
    等价原 4~6 步，但把“目标点”参数化：
      - target=(x,y)：水平移动到 (x,y)，高度保持为抬升后的高度
      - target=(x,y,z)：移动到绝对 (x,y,z)
      - target=SE3：直接到指定 SE3 位姿（若 keep_orientation=True 则仍用 T4 的朝向）
    返回: action, (T4, T5, T6)
    """
    if action is None:
        action = np.zeros(7)

    # --- 第4步：抬高 ---
    T4 = sm.SE3.Trans(0.0, 0.0, lift_height) * T_start
    pos4 = LinePositionParameter(T_start.t, T4.t)
    att4 = OneAttitudeParameter(sm.SO3(T_start.R), sm.SO3(T4.R))
    cart4 = CartesianParameter(pos4, att4)
    vel4 = QuinticVelocityParameter(time4)
    traj4 = TrajectoryParameter(cart4, vel4)
    planner4 = TrajectoryPlanner(traj4)

    # --- 第5步：到指定点（可选三种输入）---
    if isinstance(target, sm.SE3):
        T_goal = target if not keep_orientation else (sm.SE3.Trans(*target.t) * sm.SE3(sm.SO3(T4.R)))
    else:
        # tuple/list
        if len(target) == 2:
            x, y = target
            T_goal = sm.SE3.Trans(x, y, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
        elif len(target) == 3:
            x, y, z = target
            T_goal = sm.SE3.Trans(x, y, z) * sm.SE3(sm.SO3(T4.R)) if keep_orientation \
                     else sm.SE3.Trans(x, y, z)
        else:
            raise ValueError("target 需为 (x,y)、(x,y,z) 或 sm.SE3")

    T5 = T_goal
    pos5 = LinePositionParameter(T4.t, T5.t)
    att5 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cart5 = CartesianParameter(pos5, att5)
    vel5 = QuinticVelocityParameter(time5)
    traj5 = TrajectoryParameter(cart5, vel5)
    planner5 = TrajectoryPlanner(traj5)

    # --- 第6步：向下放置 ---
    T6 = sm.SE3.Trans(0.0, 0.0, -abs(descend_dist)) * T5
    pos6 = LinePositionParameter(T5.t, T6.t)
    att6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cart6 = CartesianParameter(pos6, att6)
    vel6 = QuinticVelocityParameter(time6)
    traj6 = TrajectoryParameter(cart6, vel6)
    planner6 = TrajectoryPlanner(traj6)

    # 合并执行（与原 step4to6_run 等价）
    action = _run_planners(robot, env,
                           [planner4, planner5, planner6],
                           [0.0, time4, time5, time6],
                           action, dt)

    return action, (T4, T5, T6)

def step4_make_planner(T3, time4):
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)
    return planner4, T4


def step5_make_planner(T4, time5):
    T5 = sm.SE3.Trans(1.4, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)
    return planner5, T5


def step6_make_planner(T5, time6):
    T6 = sm.SE3.Trans(0.0, 0.0, -0.25) * T5 # 放置位姿，下降25cm
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)
    return planner6, T6


def open_gripper(env, action, loops=1000, delta=0.2):
    for _ in range(loops):
        action[-1] -= delta
        action[-1] = np.max([action[-1], 0])
        env.step(action)
    return action


def step7_retract(robot, env, T6, time7, action, dt=0.002):
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    time_array = [0.0, time7]
    planner_array = [planner7]
    action = _run_planners(robot, env, planner_array, time_array, action, dt)
    return action, T7


def step8_go_home(robot, env, q8, q9, time8, action, dt=0.002):
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    time_array = [0.0, time8]
    planner_array = [planner8]
    return _run_planners(robot, env, planner_array, time_array, action, dt)

def back_to_zero(robot, env,
                  T6, q_home,
                  time7=1.0, time8=1.0,
                  action=None, dt=0.002):
    """
    第7步：抬起离开（T6 -> T7）
    第8步：回到初始（q8 -> q_home）
    返回：action, T7
    """
    if action is None:
        action = np.zeros(7)

    # 第7步：抬起
    action, T7 = step7_retract(robot, env, T6, time7, action, dt)

    # 读取当前真实关节角，确保回家轨迹连续
    q8 = robot.get_joint()

    # 第8步：回到初始
    action = step8_go_home(robot, env, q8, q_home, time8, action, dt)

    return action, T7

def prepare_grasp_context(env, gg):
    """
    抓取前的初始化，计算抓取位姿和初始关节角。
    返回:
        robot: 机器人实例
        action: 初始动作数组 (7维)
        q0: 当前关节角
        q1: 预抓取姿态
        T_wo: 物体相对于世界坐标系的抓取位姿
    """
    robot = env.robot
    action = np.zeros(7)

    # 计算相机到世界坐标的位姿
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([0.85, 0.8, 1.6])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

    # 计算物体相对于相机的位姿
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1])
    )

    # 得到物体相对于世界的位姿
    T_wo = T_wc * T_co

    # 当前关节角
    q0 = robot.get_joint()

    # 预抓取姿态（固定安全位姿）
    q1 = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0])

    return robot, action, q0, q1, T_wo, T_wc

class GraspFlow:
    """
    极薄上下文：
    - 保存 env/robot/action/q0/q1/T_wo
    - 记录关键位姿 T1~T7，便于后续步骤衔接
    - 暴露零/少参数方法，内部用你的默认参数保持行为一致
    """
    def __init__(self, env, gg, dt=0.002):
        self.env = env
        self.dt = dt
        # 复用你的初始化函数
        self.robot, self.action, self.q0, self.q1, self.T_wo, self.T_base_cam = prepare_grasp_context(env, gg)
        self._cached_img_size = None 
        # 中间位姿
        self.T1 = self.T2 = self.T3 = self.T4 = self.T5 = self.T6 = self.T7 = None
        self._target_cache = {}

    @classmethod
    def from_env(cls, env, dt=0.002):
        self = object.__new__(cls)
        self.env = env
        self.dt = dt
        self.robot = env.robot
        import numpy as np
        self.action = np.zeros(7)
        self.q0 = self.robot.get_joint()
        self.q1 = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0])

        # 用你现有的相机→基座外参（与 prepare_grasp_context 保持一致）
        import numpy as np, spatialmath as sm
        n_wc = np.array([0.0, -1.0, 0.0])
        o_wc = np.array([-1.0, 0.0, -0.5])
        t_wc = np.array([0.85, 0.8, 1.6])
        self.T_base_cam = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

        # 抓取相关位姿先置空
        self.T_wo = None
        self.T1 = self.T2 = self.T3 = self.T4 = self.T5 = self.T6 = self.T7 = None
        return self
    
    def _fetch_rgbd(self):
        """从 env 渲染一帧，用于取得当前彩色/深度和尺寸。"""
        imgs = self.env.render()
        color = imgs['img']    # RGB
        depth = imgs['depth']  # 深度
        return color, depth

    def _intrinsics_from_size(self, h, w, fovy=np.pi/4):
        """和你上游处理一致：由图像尺寸+fovy推导相机内参 (fx,fy,cx,cy)。"""
        focal = h / (2.0 * np.tan(fovy / 2.0))
        cx = w / 2.0
        cy = h / 2.0
        return (focal, focal, cx, cy)

    # 1~3步：移至抓取位姿（预抓取→接近→到达）
    def move_to_grasp_pose(self, time1=1.0, time2=1.0, time3=1.0, approach_offset=(-0.1, 0.0, 0.0)):
        self.action, self.T2, self.T3 = move_to_grasp_pose(
            self.robot, self.env,
            q0=self.q0, q1=self.q1, T_wo=self.T_wo,
            time1=time1, time2=time2, time3=time3,
            action=self.action, dt=self.dt,
            approach_offset=approach_offset
        )
        return self

    # 关夹爪
    def close_gripper(self, loops=1000, delta=0.2):
        self.action = close_gripper(self.env, self.action, loops=loops, delta=delta)
        return self

    # 4~6步：移动到指定点并下放 —— 仅需传目标点
    def move_to_point(self, target_place,
                      time4=1.0, time5=1.0, time6=1.0,
                      lift_height=0.35,
                      descend_dist=0.25,
                      keep_orientation=True,
                      T_start=None,
                      depth_scale=1.0,   # 若深度是毫米，改为 0.001；仿真通常1.0(米)
                      z_offset=0.4):    # 悬停高度偏置，避免刮擦
        # 若未给起点，用当前真实末端位姿
        if T_start is None:
            T_start = self.robot.get_cartesian()

        # ---- 自动识别 target_place 是像素还是基座 ----
        tgt = target_place
        as_pixel = (
            isinstance(tgt, (tuple, list)) and
            len(tgt) == 2 and
            all(isinstance(v, (int, np.integer)) for v in tgt)
        )

        if as_pixel:
            # 1) 取一帧图像，得到图像尺寸
            color, depth = self._fetch_rgbd()
            h, w = depth.shape[:2]
            # 2) 内参（与上游一致）
            if not self._cached_img_size or self._cached_img_size != (h, w):
                self.K = self._intrinsics_from_size(h, w)  # (fx, fy, cx, cy)
                self._cached_img_size = (h, w)
            # 3) 像素→深度→基座
            u, v = int(tgt[0]), int(tgt[1])
            Zm = _depth_at_median(depth, u, v, win=3, depth_scale=depth_scale)
            xb, yb, zb = _pixel2base_xyz(u, v, Zm, self.K, self.T_base_cam)
            target_xyz = (xb, yb, zb + float(z_offset))
        else:
            # 已是基座输入，仍可使用 z_offset 调整高度
            if isinstance(tgt, (tuple, list)) and len(tgt) == 3:
                x, y, z = tgt
                target_xyz = (x, y, z + float(z_offset))  # ★ 加上传入的 z_offset
            elif isinstance(tgt, (tuple, list)) and len(tgt) == 2:
                # 仅给了 (x, y)，保持当前抬升高度
                x, y = tgt
                if z_offset != 0.0:
                    target_xyz = (x, y, z + float(z_offset))
                else:
                    target_xyz = tgt 
            elif isinstance(tgt, sm.SE3):
                # 若是完整位姿，可以选择是否在位姿上加高
                T_goal = sm.SE3.Trans(0, 0, float(z_offset)) * tgt
                target_xyz = T_goal
            else:
                raise ValueError("target_place 必须为 (x,y)、(x,y,z) 或 sm.SE3")


        # 调用你原来的 move_to_point（不改签名/实现）
        self.action, (self.T4, self.T5, self.T6) = move_to_point(
            self.robot, self.env,
            T_start=T_start,
            target=target_xyz,
            time4=time4, time5=time5, time6=time6,
            action=self.action, dt=self.dt,
            lift_height=lift_height, descend_dist=descend_dist,
            keep_orientation=keep_orientation
        )
        # 下一段起点沿用
        self.T3 = self.T6
        return self


    # 开夹爪
    def open_gripper(self, loops=1000, delta=0.2):
        self.action = open_gripper(self.env, self.action, loops=loops, delta=delta)
        return self

    # 7+8步：抬起离开并回到初始
    def back_to_zero(self, time7=1.0, time8=1.0):
        # 兜底：如果 T6 为空，优先用当前末端姿态；再退而求其次用 T3
        T6_fallback = self.T6
        if T6_fallback is None:
            try:
                T6_fallback = self.robot.get_cartesian()
                print("[WARN] self.T6 为空，使用当前末端位姿作为 T6")
            except Exception:
                if self.T3 is not None:
                    T6_fallback = self.T3
                    print("[WARN] self.T6 为空，使用 T3 作为 T6")
                else:
                    raise RuntimeError("没有可用的 T6/T3，无法执行 back_to_zero()")

        self.action, self.T7 = back_to_zero(
            self.robot, self.env,
            T6=T6_fallback, q_home=self.q0,
            time7=time7, time8=time8,
            action=self.action, dt=self.dt
        )
        return self

def execute_grasp(env, gg,
                  # 指定放置目标：可给 (x,y)，或 (x,y,z)，或 sm.SE3
                  target_place=(1.4, 0.3),
                  # 每段时长（与原默认一致）
                  t1=1.0, t2=1.0, t3=1.0, t4=1.0, t5=1.0, t6=1.0, t7=1.0, t8=1.0,
                  dt=0.002):
    """
    示例流程（按你需求把动作函数串起来）：
      1) 移至抓取位姿（1~3步）
      2) 关闭夹爪
      3) 移动到“指定点”并下放（4~6步，目标可指定）
      4) 打开夹爪
      5) 7、8 步骤不变：抬起离开、回到初始
    """
    robot = env.robot

    # 0) 计算抓取位姿 T_wo（保持你原逻辑）
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([0.85, 0.8, 1.6])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1])
    )
    T_wo = T_wc * T_co

    # 初始 action / 姿态
    action = np.zeros(7)
    q0 = robot.get_joint()
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

    # 1) 移至抓取位姿（1~3步）
    action, T2, T3 = move_to_grasp_pose(
        robot, env, q0, q1, T_wo,
        time1=t1, time2=t2, time3=t3,
        action=action, dt=dt,
        approach_offset=(-0.1, 0.0, 0.0)  # 与原默认一致
    )

    # 2) 关闭夹爪
    action = close_gripper(env, action, loops=1000, delta=0.2)

    # 3) 移动到“指定点”并下放（4~6步）
    action, (T4, T5, T6) = move_to_point(
        robot, env,
        T_start=T3,                # 从抓取位姿出发
        target=target_place,       # 这里可传 (x,y)/(x,y,z)/SE3
        time4=t4, time5=t5, time6=t6,
        action=action, dt=dt,
        lift_height=0.3,           # 与原默认一致
        descend_dist=0.25,          # 与原默认一致
        keep_orientation=True      # 保持末端朝向不变
    )

    # 4) 打开夹爪
    action = open_gripper(env, action, loops=1000, delta=0.2)

    # 5) 第 7、8 步不变：抬起离开 + 回到初始
    action, T7 = back_to_zero(robot, env, T6, q_home=q0, time7=t7, time8=t8, action=action, dt=dt)

    return True


