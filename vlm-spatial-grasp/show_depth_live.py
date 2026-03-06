# show_depth_live.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np

# =========================
# 1) 深度可视化工具函数
# =========================
def depth_to_colormap(depth_img,
                      near=0.20, far=1.20,
                      depth_unit="m",   # "m" or "mm"
                      invalid_as_far=True):
    """
    depth_img: HxW, float32(米) / uint16(毫米) / float32(任意)
    返回: depth_color(BGR), depth_u8(灰度)
    """
    d = depth_img

    # 统一到 float32 米
    if d.dtype == np.uint16:
        d = d.astype(np.float32)
        if depth_unit == "mm":
            d *= 0.001  # mm -> m
    else:
        d = d.astype(np.float32)

    # 无效值处理
    mask_invalid = (~np.isfinite(d)) | (d <= 0)
    if invalid_as_far:
        d[mask_invalid] = far
    else:
        d[mask_invalid] = near

    # 裁剪 + 归一化
    d = np.clip(d, near, far)
    norm = (d - near) / (far - near + 1e-6)  # 0..1
    gray = (255.0 * (1.0 - norm)).astype(np.uint8)  # 近亮远暗

    # 伪彩色
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return color, gray


# =========================
# 2) 获取一帧图像：接入你的项目
#    固定假设：env.render()["img"] 是 RGB
# =========================
def get_frame_from_your_project():
    """
    返回: (color_bgr_uint8, depth_float32)
    自动热重载：当 scene / include xml 发生变化时重建 env
    """
    import os
    import sys
    import time
    import numpy as np
    import cv2

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    mg_path = os.path.join(ROOT_DIR, "manipulator_grasp")
    if mg_path not in sys.path:
        sys.path.insert(0, mg_path)

    from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

    # -------- 热重载：监控 XML 的最新修改时间 --------
    # 1) scene.xml 路径（按你 env.reset() 里的写法推出来）
    scene_path = os.path.join(ROOT_DIR, "manipulator_grasp", "assets", "scenes", "scene.xml")
    scenes_dir = os.path.dirname(scene_path)

    def latest_xml_mtime(folder):
        latest = 0.0
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.endswith(".xml"):
                    p = os.path.join(root, fn)
                    try:
                        latest = max(latest, os.path.getmtime(p))
                    except FileNotFoundError:
                        pass
        return latest

    global _DEPTH_ENV, _XML_MTIME, _LAST_CHECK
    if "_DEPTH_ENV" not in globals():
        _DEPTH_ENV = None
    if "_XML_MTIME" not in globals():
        _XML_MTIME = None
    if "_LAST_CHECK" not in globals():
        _LAST_CHECK = 0.0

    # 每 0.2s 检查一次文件变化（避免每帧都 os.walk）
    now = time.time()
    if now - _LAST_CHECK > 0.2:
        _LAST_CHECK = now
        cur_mtime = latest_xml_mtime(scenes_dir)  # ✅ 能覆盖 include 文件的修改
        # 如果你只想盯 scene.xml，本行改成：cur_mtime = os.path.getmtime(scene_path)

        if _XML_MTIME is None:
            _XML_MTIME = cur_mtime

        # 触发重载
        if cur_mtime != _XML_MTIME and _DEPTH_ENV is not None:
            print("[HOTRELOAD] XML changed, recreating env...")
            try:
                _DEPTH_ENV.close()
            except Exception as e:
                print("[HOTRELOAD] close() failed:", e)
            _DEPTH_ENV = None
            _XML_MTIME = cur_mtime

    # -------- 初始化 / 重建 env --------
    if _DEPTH_ENV is None:
        _DEPTH_ENV = UR5GraspEnv()
        _DEPTH_ENV.reset()
        for _ in range(50):
            _DEPTH_ENV.step()

    # 走两步，拿到稳定观测
    for _ in range(2):
        _DEPTH_ENV.step()

    imgs = _DEPTH_ENV.render()
    if not isinstance(imgs, dict) or "img" not in imgs or "depth" not in imgs:
        raise RuntimeError(f"env.render() 返回不包含 img/depth，keys={list(imgs.keys()) if isinstance(imgs, dict) else type(imgs)}")

    color_rgb = np.asarray(imgs["img"])
    depth = np.asarray(imgs["depth"]).astype(np.float32)

    if color_rgb.ndim == 3 and color_rgb.shape[2] == 3:
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    else:
        raise RuntimeError(f"期望 img 为 HxWx3 RGB，但得到 shape={getattr(color_rgb, 'shape', None)}")

    return color_bgr, depth

# =========================
# 3) 主循环：实时显示
# =========================
def main():
    near, far = 0.25, 1.50
    depth_unit = "m"   # 如果你的 depth 是 uint16 毫米，改成 "mm"
    show_rgb = True
    printed_info = False

    print("按键说明：")
    print("  [ / ]  : far -/+ 0.1m")
    print("  - / =  : near -/+ 0.05m")
    print("  r      : 切换是否显示RGB")
    print("  ESC    : 退出")

    while True:
        color_bgr, depth_img = get_frame_from_your_project()

        if not printed_info:
            print(f"[INFO] color_bgr shape={color_bgr.shape} dtype={color_bgr.dtype}")
            print(f"[INFO] depth_img  shape={depth_img.shape} dtype={depth_img.dtype} "
                  f"min={np.nanmin(depth_img):.4f} max={np.nanmax(depth_img):.4f}")
            printed_info = True

        depth_color, _ = depth_to_colormap(depth_img, near=near, far=far, depth_unit=depth_unit)
        cv2.imshow("Depth (ColorMap)", depth_color)

        if show_rgb:
            cv2.imshow("RGB", color_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            show_rgb = not show_rgb
        elif key == ord('['):
            far = max(near + 0.05, far - 0.1)
            print(f"[CFG] near={near:.2f}m far={far:.2f}m")
        elif key == ord(']'):
            far = far + 0.1
            print(f"[CFG] near={near:.2f}m far={far:.2f}m")
        elif key == ord('-'):
            near = max(0.01, near - 0.05)
            print(f"[CFG] near={near:.2f}m far={far:.2f}m")
        elif key == ord('='):
            near = min(far - 0.05, near + 0.05)
            print(f"[CFG] near={near:.2f}m far={far:.2f}m")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
