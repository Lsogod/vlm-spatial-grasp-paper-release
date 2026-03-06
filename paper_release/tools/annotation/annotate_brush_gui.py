# annotate_brush_gui.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QComboBox, QTextEdit, QSlider,
    QRadioButton, QCheckBox, QShortcut, QButtonGroup, 
    QListWidget, QListWidgetItem, QAbstractItemView, QProgressDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QPen, QColor, QKeySequence

class NullWriter:
    def write(self, text): pass
    def flush(self): pass

if sys.stdout is None: sys.stdout = NullWriter()
if sys.stderr is None: sys.stderr = NullWriter()


# ========== 配置 ==========

ACTION_POOL = [
    "move_to_grasp_pose",
    "close_gripper",
    "move_to_point",
    "open_gripper",
    "back_to_zero",
]

ACTION_DESCRIPTIONS = {
    "move_to_grasp_pose": "移动到抓取姿态",
    "close_gripper":      "闭合夹爪",
    "move_to_point":      "移动到指定点",
    "open_gripper":       "打开夹爪",
    "back_to_zero":       "回到初始位姿",
}

TASK_TYPE_MAP = {
    "抓取任务": "pick",
    "抓取放置任务": "place",
    "无特定对象/模糊意图": "ambiguous"
}

TASK_SPECIFIC_DEFAULTS = {
    "pick": ["move_to_grasp_pose", "close_gripper", "back_to_zero"],
}

BRUSH_RADIUS = 30
BRUSH_ALPHA = 0.5

# 颜色配置 (BGR)
COLOR_PREVIEW = (255, 150, 0)   # 预览色
COLOR_GRASP   = (0, 255, 0)     # 绿
COLOR_PLACE   = (0, 0, 255)     # 红
COLOR_HIGHLIGHT = (0, 255, 255) # 黄色 (用于高亮选中区域)


def point_in_mask(x, y, mask):
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    return mask[int(y), int(x)] > 0


def choose_nearest(candidates, x, y):
    if not candidates: return None
    return min(
        candidates,
        key=lambda it: (it[1]["centroid"][0] - x) ** 2 + (it[1]["centroid"][1] - y) ** 2
    )


def create_brush_mask(h, w, x, y, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(radius), 1, thickness=-1)
    return mask.astype(bool)


def redraw_image(img_base, brush_regions, masks,
                 current_pos=None, current_mask_idx=None, current_brush_type="grasp",
                 highlight_indices=None):
    """
    重绘图像
    highlight_indices: list of int, 需要高亮的 brush_regions 索引列表
    """
    h, w = img_base.shape[:2]
    img_vis = img_base.copy()

    # 1. 绘制已保存区域
    for i, region in enumerate(brush_regions):
        painted = region["painted_area"]
        if painted is None: continue
        
        r_type = region.get("type", "grasp")
        color = COLOR_PLACE if r_type == "place" else COLOR_GRASP

        overlay = img_vis.copy()
        overlay[painted] = (
            overlay[painted] * (1 - BRUSH_ALPHA) + np.array(color, dtype=np.float32) * BRUSH_ALPHA
        ).astype(np.uint8)
        img_vis = overlay

        # 高亮选中区域 (根据列表索引高亮)
        if highlight_indices and i in highlight_indices:
            mask_u8 = painted.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 画两层轮廓确保清晰
            cv2.drawContours(img_vis, contours, -1, (0,0,0), 5) # 黑底
            cv2.drawContours(img_vis, contours, -1, COLOR_HIGHLIGHT, 2) # 黄芯

    # 2. 绘制当前预览
    if current_pos is not None and current_mask_idx is not None:
        x, y = current_pos
        brush = create_brush_mask(h, w, x, y, BRUSH_RADIUS)
        mask_bin = (masks[current_mask_idx] > 0).astype(bool)
        preview = brush & mask_bin

        if current_brush_type == "place":
            preview_color = COLOR_PLACE
        elif current_brush_type == "grasp":
            preview_color = COLOR_GRASP
        else:
            preview_color = COLOR_PREVIEW 

        overlay = img_vis.copy()
        overlay[preview] = (
            overlay[preview] * (1 - BRUSH_ALPHA) + np.array(preview_color, dtype=np.float32) * BRUSH_ALPHA
        ).astype(np.uint8)
        img_vis = overlay

    return img_vis


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent():
            self.parent().on_mouse_press(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.parent():
            self.parent().on_mouse_move(event)
        else:
            if self.parent():
                self.parent().on_mouse_move(event, painting=False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.parent():
            self.parent().on_mouse_release(event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brush 标注 GUI")

        # 状态
        self.npz_path = None
        self.folder_path = None
        self.folder_metadata = [] # 存储所有文件的元数据 {path, types, annotated}
        self.content = None       
        self.image_rgb = None     
        self.base_bgr = None      
        self.vis_bgr = None       
        self.instance_masks = None    
        self.results = []         
        self.H, self.W = 0, 0
        self.trajectories = []
        self.brush_regions = []       
        self.all_painted_mask = None 
        self.current_region = None
        self.is_painting = False
        self.current_mask_idx = None
        self.hover_mask_idx = None
        self.mode = "grasp"
        self.selected_region_indices = set() 

        self.action_buttons = []
        self.selected_actions = list(ACTION_POOL)
        self.brush_radius = BRUSH_RADIUS

        # ========== 布局构建 ==========
        left_layout = QVBoxLayout()

        # 1. 文件夹选择与筛选
        folder_group = QGroupBox("文件导航")
        folder_layout = QVBoxLayout()
        
        # 行1: 选择文件夹按钮 + 路径显示
        row1 = QHBoxLayout()
        self.btn_open_folder = QPushButton("📂 选择文件夹")
        self.btn_open_folder.clicked.connect(self.choose_folder)
        row1.addWidget(self.btn_open_folder)
        self.lbl_folder = QLabel("未选择")
        self.lbl_folder.setStyleSheet("color: gray;")
        row1.addWidget(self.lbl_folder)
        row1.addStretch()
        folder_layout.addLayout(row1)

        # 行2: 筛选器
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("🔍 筛选包含类型:"))
        self.combo_filter_type = QComboBox()
        self.combo_filter_type.addItem("显示全部", "all")
        self.combo_filter_type.addItem("包含 [Pick] 任务", "pick")
        self.combo_filter_type.addItem("包含 [Place] 任务", "place")
        self.combo_filter_type.addItem("包含 [Ambiguous] 任务", "ambiguous")
        self.combo_filter_type.addItem("只看 [已标注] 文件", "annotated")
        self.combo_filter_type.addItem("只看 [未标注] 文件", "unannotated")
        self.combo_filter_type.currentIndexChanged.connect(self.update_file_list_by_filter)
        row2.addWidget(self.combo_filter_type)
        folder_layout.addLayout(row2)

        # 行3: 上一张/下一张 + 文件下拉框
        row3 = QHBoxLayout()
        self.btn_prev_file = QPushButton("<< 上一张")
        self.btn_prev_file.setShortcut("A") # 快捷键 A
        self.btn_prev_file.setToolTip("快捷键: A")
        self.btn_prev_file.clicked.connect(self.on_prev_file_clicked)
        self.btn_prev_file.setMaximumWidth(80) # [修改] 限制按钮宽度
        row3.addWidget(self.btn_prev_file)

        self.combo_files = QComboBox()
        self.combo_files.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        
        # [修改] 设置最小宽度，防止被压缩
        self.combo_files.setMinimumWidth(250) 
        # [修改] 设置下拉列表视图的最小宽度，确保展开时能看清长文件名
        self.combo_files.view().setMinimumWidth(400)
        
        self.combo_files.currentIndexChanged.connect(self.on_file_selected)
        row3.addWidget(self.combo_files, stretch=1)

        self.btn_next_file = QPushButton("下一张 >>")
        self.btn_next_file.setShortcut("D") # 快捷键 D
        self.btn_next_file.setToolTip("快捷键: D")
        self.btn_next_file.clicked.connect(self.on_next_file_clicked)
        self.btn_next_file.setMaximumWidth(80) # [修改] 限制按钮宽度
        row3.addWidget(self.btn_next_file)
        
        folder_layout.addLayout(row3)
        
        self.lbl_file_count = QLabel("当前列表: 0/0")
        self.lbl_file_count.setAlignment(Qt.AlignCenter)
        folder_layout.addWidget(self.lbl_file_count)

        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)

        # 2. 任务概览
        left_layout.addWidget(QLabel("当前文件已有任务:"))
        self.traj_view = QTextEdit()
        self.traj_view.setReadOnly(True)
        self.traj_view.setMinimumHeight(100)
        self.traj_view.setStyleSheet("font-family: Consolas; font-size: 11px;") 
        left_layout.addWidget(self.traj_view)
        
        # 任务操作
        traj_op_layout = QHBoxLayout()
        self.combo_traj_select = QComboBox()
        traj_op_layout.addWidget(self.combo_traj_select)
        
        self.btn_preview_traj = QPushButton("预览/加载")
        self.btn_preview_traj.clicked.connect(self.on_preview_traj_clicked)
        traj_op_layout.addWidget(self.btn_preview_traj)
        
        self.btn_delete_traj = QPushButton("删除")
        self.btn_delete_traj.clicked.connect(self.on_delete_traj_clicked)
        traj_op_layout.addWidget(self.btn_delete_traj)
        
        self.btn_overwrite_traj = QPushButton("重写")
        self.btn_overwrite_traj.clicked.connect(self.on_overwrite_traj_clicked)
        traj_op_layout.addWidget(self.btn_overwrite_traj)
        left_layout.addLayout(traj_op_layout)

        # 3. 任务属性
        left_layout.addWidget(QLabel("Instruction:"))
        self.edit_instruction = QLineEdit()
        left_layout.addWidget(self.edit_instruction)

        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel("任务类型:"))
        self.combo_task_type = QComboBox()
        self.combo_task_type.addItems(list(TASK_TYPE_MAP.keys()))
        self.combo_task_type.currentIndexChanged.connect(self.on_task_type_changed)
        task_type_layout.addWidget(self.combo_task_type)
        self.chk_lock_default = QCheckBox("设为默认")
        self.chk_lock_default.toggled.connect(self.on_lock_default_toggled)
        task_type_layout.addWidget(self.chk_lock_default)
        left_layout.addLayout(task_type_layout)

        # 4. 区域管理列表
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("已绘区域列表 (同Mask>=2自动全选):"))
        self.region_list_widget = QListWidget()
        self.region_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection) 
        self.region_list_widget.setMinimumHeight(150)
        self.region_list_widget.itemSelectionChanged.connect(self.on_region_selection_changed)
        left_layout.addWidget(self.region_list_widget)

        # 区域操作按钮
        region_btn_layout = QHBoxLayout()
        self.btn_switch_selected = QPushButton("切换类型 (绿<->红)")
        self.btn_switch_selected.setStyleSheet("font-weight: bold; color: blue;")
        self.btn_switch_selected.clicked.connect(self.on_switch_selected_clicked)
        region_btn_layout.addWidget(self.btn_switch_selected)
        
        self.btn_delete_selected_region = QPushButton("删除选中")
        self.btn_delete_selected_region.clicked.connect(self.on_delete_selected_region_clicked)
        region_btn_layout.addWidget(self.btn_delete_selected_region)
        left_layout.addLayout(region_btn_layout)

        # 5. 画笔与模式
        brush_layout = QHBoxLayout()
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 100)
        self.brush_slider.setValue(BRUSH_RADIUS)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        brush_layout.addWidget(QLabel("笔刷:"))
        brush_layout.addWidget(self.brush_slider)
        self.brush_size_label = QLabel(str(BRUSH_RADIUS))
        brush_layout.addWidget(self.brush_size_label)
        left_layout.addLayout(brush_layout)

        mode_layout = QHBoxLayout()
        self.mode_bg = QButtonGroup(self) 
        self.radio_grasp = QRadioButton("抓取 (Grasp)")
        self.radio_grasp.setStyleSheet("color: green; font-weight: bold;")
        self.radio_grasp.setChecked(True)
        self.mode_bg.addButton(self.radio_grasp)
        self.radio_place = QRadioButton("放置 (Place)")
        self.radio_place.setStyleSheet("color: red; font-weight: bold;")
        self.mode_bg.addButton(self.radio_place)
        self.radio_eraser = QRadioButton("橡皮擦")
        self.mode_bg.addButton(self.radio_eraser)
        self.radio_grasp.toggled.connect(self.on_mode_changed)
        self.radio_place.toggled.connect(self.on_mode_changed)
        self.radio_eraser.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_grasp)
        mode_layout.addWidget(self.radio_place)
        mode_layout.addWidget(self.radio_eraser)
        left_layout.addLayout(mode_layout)

        # 6. 动作序列
        actions_group = QGroupBox("动作选择")
        act_layout = QVBoxLayout()
        grid = QGridLayout()
        for row, act in enumerate(ACTION_POOL):
            btn = QPushButton(act)
            btn.clicked.connect(self.on_action_button_clicked)
            self.action_buttons.append(btn)
            grid.addWidget(btn, row, 0)
            grid.addWidget(QLabel(ACTION_DESCRIPTIONS.get(act, "")), row, 1)
        act_layout.addLayout(grid)
        self.actions_view = QLabel()
        self.actions_view.setWordWrap(True)
        self.actions_view.setStyleSheet("border: 1px solid gray; padding: 3px;")
        act_layout.addWidget(QLabel("当前顺序:"))
        act_layout.addWidget(self.actions_view)
        
        act_ctrl = QHBoxLayout()
        b_clr = QPushButton("清空")
        b_clr.clicked.connect(self.on_clear_actions_clicked)
        act_ctrl.addWidget(b_clr)
        b_def = QPushButton("重置默认")
        b_def.clicked.connect(self.on_default_actions_clicked)
        act_ctrl.addWidget(b_def)
        act_layout.addLayout(act_ctrl)
        actions_group.setLayout(act_layout)
        left_layout.addWidget(actions_group)

        # 7. 全局操作
        self.btn_start_traj = QPushButton("开始新任务 (Ctrl+N)")
        self.btn_start_traj.clicked.connect(self.start_new_trajectory)
        left_layout.addWidget(self.btn_start_traj)

        self.btn_finish_traj = QPushButton("保存任务 (Ctrl+Enter)")
        self.btn_finish_traj.clicked.connect(self.finish_trajectory)
        left_layout.addWidget(self.btn_finish_traj)

        self.btn_save_npz = QPushButton("写回 NPZ (Ctrl+S)")
        self.btn_save_npz.clicked.connect(self.save_npz)
        left_layout.addWidget(self.btn_save_npz)
        left_layout.addStretch()

        # 主布局
        self.image_label = ImageLabel(self)
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=0)
        main_layout.addWidget(self.image_label, stretch=1)
        self.setLayout(main_layout)

        # 快捷键
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_npz)
        QShortcut(QKeySequence("Ctrl+N"), self).activated.connect(self.start_new_trajectory)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.finish_trajectory)
        
        self.update_cursor()
        self.on_task_type_changed(0)

    # ========== [新增逻辑] 文件夹扫描与筛选 ==========

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", ".")
        if not folder: return
        
        self.folder_path = folder
        self.lbl_folder.setText(os.path.basename(folder))
        
        # 扫描文件夹
        self.scan_folder_metadata()
        
        # 默认重置筛选器
        self.combo_filter_type.setCurrentIndex(0)
        self.update_file_list_by_filter()

    def scan_folder_metadata(self):
        """扫描所有 npz 文件，记录包含的任务类型，用于筛选"""
        if not self.folder_path: return
        
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.npz')]
        files.sort()
        
        self.folder_metadata = []
        
        progress = QProgressDialog("正在扫描文件元数据...", "取消", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        
        for i, fname in enumerate(files):
            if progress.wasCanceled(): break
            progress.setValue(i)
            
            full_path = os.path.join(self.folder_path, fname)
            item_data = {
                "path": full_path,
                "name": fname,
                "types": set(),
                "annotated": False
            }
            
            try:
                # 尝试轻量读取
                d = np.load(full_path, allow_pickle=True)
                if "trajectories" in d.files:
                    raw = d["trajectories"]
                    if isinstance(raw, np.ndarray):
                        s = raw.item() if raw.ndim == 0 else raw.reshape(-1)[0]
                    else: s = raw
                    
                    try:
                        trajs = json.loads(s)
                        if trajs and isinstance(trajs, list):
                            item_data["annotated"] = True
                            for t in trajs:
                                item_data["types"].add(t.get("task_type", "unknown"))
                    except:
                        pass
            except:
                pass
                
            self.folder_metadata.append(item_data)
        
        progress.setValue(len(files))

    def update_file_list_by_filter(self):
            """根据下拉框筛选条件，重新填充 combo_files"""
            if not self.folder_metadata: return
            
            # [新增] 1. 记录当前选中的文件路径
            current_path = None
            if self.combo_files.count() > 0:
                current_path = self.combo_files.currentData()
            
            filter_mode = self.combo_filter_type.currentData() # "all", "pick", "place", "annotated"...
            
            filtered_files = []
            
            for item in self.folder_metadata:
                match = False
                if filter_mode == "all":
                    match = True
                elif filter_mode == "annotated":
                    match = item["annotated"]
                elif filter_mode == "unannotated":
                    match = not item["annotated"]
                else:
                    # 按任务类型筛选 (pick, place, etc.)
                    if filter_mode in item["types"]:
                        match = True
                
                if match:
                    filtered_files.append(item)
            
            # 刷新下拉框
            self.combo_files.blockSignals(True)
            self.combo_files.clear()
            
            target_index = 0 # 默认选中第0个
            
            for i, item in enumerate(filtered_files):
                # 标记已标注
                prefix = "[已标] " if item["annotated"] else "[未标] "
                self.combo_files.addItem(prefix + item["name"], item["path"])
                
                # [新增] 2. 如果这个文件就是刚才选中的文件，记录它的新索引
                if current_path and item["path"] == current_path:
                    target_index = i
            
            self.combo_files.blockSignals(False)
            
            # 更新计数标签
            total = len(self.folder_metadata)
            current_show = len(filtered_files)
            self.lbl_file_count.setText(f"当前列表: {current_show}/{total}")
            
            # [新增] 3. 恢复选中状态
            if self.combo_files.count() > 0:
                self.combo_files.setCurrentIndex(target_index)
                # 只有当索引确实改变（或者这是初始化加载）时，才触发加载逻辑
                # 但为了保险起见，如果是保存后刷新，我们通常不需要重新加载文件内容(因为内容没变)
                # 不过为了逻辑简单，这里重新触发一次加载也无妨
                if self.combo_files.itemData(target_index) != self.npz_path:
                    self.on_file_selected(target_index)
            else:
                # 清空界面
                self.lbl_file_count.setText(f"当前列表: 0/{total} (无匹配)")
                self.image_label.clear()
                self.traj_view.clear()
                self.region_list_widget.clear()
    # ========== [新增逻辑] 翻页按钮 ==========

    def on_prev_file_clicked(self):
        count = self.combo_files.count()
        if count == 0: return
        
        curr = self.combo_files.currentIndex()
        new_idx = curr - 1
        if new_idx < 0:
            new_idx = count - 1 # 循环到最后
        
        self.combo_files.setCurrentIndex(new_idx)

    def on_next_file_clicked(self):
        count = self.combo_files.count()
        if count == 0: return
        
        curr = self.combo_files.currentIndex()
        new_idx = curr + 1
        if new_idx >= count:
            new_idx = 0 # 循环到开始
        
        self.combo_files.setCurrentIndex(new_idx)

    # ========== 区域列表管理 ==========

    def update_region_list_ui(self):
        self.region_list_widget.blockSignals(True)
        # 通过 list_widget 获取 row
        current_rows = [self.region_list_widget.row(item) for item in self.region_list_widget.selectedItems()]
        
        self.region_list_widget.clear()
        for i, r in enumerate(self.brush_regions):
            rtype = r.get("type", "grasp")
            rid = r.get("id", -1)
            txt = f"[{i}] Grasp (绿色) - Mask {rid}" if rtype == "grasp" else f"[{i}] Place (红色) - Mask {rid}"
            color = QColor("darkgreen") if rtype == "grasp" else QColor("darkred")
            
            item = QListWidgetItem(txt)
            item.setForeground(color)
            item.setData(Qt.UserRole, i) 
            self.region_list_widget.addItem(item)
            
        for row in current_rows:
            if row < self.region_list_widget.count():
                self.region_list_widget.item(row).setSelected(True)
        
        self.region_list_widget.blockSignals(False)

    def on_region_selection_changed(self):
        selected_items = self.region_list_widget.selectedItems()
        if not selected_items:
            self.selected_region_indices.clear()
            self.refresh_view()
            return
        
        # 智能全选同Mask逻辑
        mask_counts = {}
        for item in selected_items:
            row = self.region_list_widget.row(item)
            if row < len(self.brush_regions):
                midx = self.brush_regions[row]["mask_index"]
                mask_counts[midx] = mask_counts.get(midx, 0) + 1
        
        target_masks = {midx for midx, count in mask_counts.items() if count >= 2}
        
        if target_masks:
            self.region_list_widget.blockSignals(True)
            for i in range(self.region_list_widget.count()):
                if i < len(self.brush_regions):
                    if self.brush_regions[i]["mask_index"] in target_masks:
                        self.region_list_widget.item(i).setSelected(True)
            self.region_list_widget.blockSignals(False)
            selected_items = self.region_list_widget.selectedItems()

        self.selected_region_indices.clear()
        for item in selected_items:
            row = self.region_list_widget.row(item)
            self.selected_region_indices.add(row)
        
        self.refresh_view()

    def on_switch_selected_clicked(self):
        selected_rows = [self.region_list_widget.row(item) for item in self.region_list_widget.selectedItems()]
        if not selected_rows: return

        count = 0
        for row in selected_rows:
            if row < len(self.brush_regions):
                r = self.brush_regions[row]
                r["type"] = "place" if r.get("type", "grasp") == "grasp" else "grasp"
                count += 1
            
        self.update_region_list_ui()
        self.refresh_view()

    def on_delete_selected_region_clicked(self):
        selected_rows = [self.region_list_widget.row(item) for item in self.region_list_widget.selectedItems()]
        if not selected_rows: return
        selected_rows.sort(reverse=True)
        
        for row in selected_rows:
            if row < len(self.brush_regions):
                self.brush_regions.pop(row)
            
        self.all_painted_mask = np.zeros((self.H, self.W), dtype=bool)
        for r in self.brush_regions:
            self.all_painted_mask |= r["painted_area"]
            
        self.selected_region_indices.clear()
        self.update_region_list_ui()
        self.refresh_view()

    # ========== 任务列表显示 ==========

    def update_traj_view_text(self):
            self.combo_traj_select.blockSignals(True)
            self.combo_traj_select.clear()
            self.combo_traj_select.addItems([f"任务 {i}" for i in range(len(self.trajectories))])
            self.combo_traj_select.blockSignals(False)

            if not self.trajectories:
                self.traj_view.setPlainText("（当前文件暂无标注）")
                return

            lines = []
            for i, traj in enumerate(self.trajectories):
                instr = traj.get("instruction", "")
                task_type = traj.get("task_type", "未知类型")
                
                # --- 获取动作序列 ---
                action_data = traj.get("action", {})
                act_list = action_data.get("actions", []) if isinstance(action_data, dict) else []
                act_str = " -> ".join(act_list) if act_list else "(无动作)"

                # --- 统计区域 ---
                n_grasp = len(traj.get("grasp_regions", []))
                n_place = len(traj.get("place_regions", []))
                # [修改] 移除了 n_old 的统计

                lines.append(f"[{i}] {task_type}")
                lines.append(f"   Instr:   {instr}")
                lines.append(f"   Actions: {act_str}")
                # [修改] 界面上不再显示 Old，只显示 G 和 P
                lines.append(f"   Regions: G={n_grasp}, P={n_place}")
                lines.append("-" * 40)
            
            self.traj_view.setPlainText("\n".join(lines))
    # ========== 加载与保存 ==========

    def on_preview_traj_clicked(self):
        if self.base_bgr is None or not self.trajectories: return
        idx = self.combo_traj_select.currentIndex()
        if idx < 0: return

        traj = self.trajectories[idx]
        self.selected_actions = list(traj.get("action", {}).get("actions", []))
        self.update_action_sequence_view()
        self.edit_instruction.setText(traj.get("instruction", ""))
        
        saved_type = traj.get("task_type", "")
        for k, v in TASK_TYPE_MAP.items():
            if v == saved_type:
                self.combo_task_type.blockSignals(True)
                self.combo_task_type.setCurrentText(k)
                self.combo_task_type.blockSignals(False)
                break
        
        self.brush_regions = []
        self.all_painted_mask = np.zeros((self.H, self.W), dtype=bool)
        H, W = self.H, self.W
        
        def load_list(r_list, r_type):
            for r in r_list:
                coords = r.get("painted_coords", [])
                mask = np.zeros((H, W), dtype=bool)
                for item in coords:
                    if len(item) == 2: mask[int(item[0]), int(item[1])] = True
                self.brush_regions.append({
                    "id": int(r.get("id", 0)), 
                    "mask_index": int(r.get("mask_index", 0)),
                    "painted_area": mask,
                    "type": r_type
                })
                self.all_painted_mask |= mask

        load_list(traj.get("grasp_regions", []), "grasp")
        load_list(traj.get("place_regions", []), "place")
        if "brush_regions" in traj:
            load_list(traj.get("brush_regions", []), "grasp")

        self.update_region_list_ui()
        self.refresh_view()

    # ========== 常规逻辑 ==========

    def refresh_view(self, preview_pos=None, preview_mask_idx=None):
        if self.base_bgr is None: return
        self.vis_bgr = redraw_image(
            self.base_bgr, self.brush_regions, self.instance_masks,
            preview_pos, preview_mask_idx, self.mode,
            highlight_indices=list(self.selected_region_indices)
        )
        self.image_label.setPixmap(self.cvimg_to_qpixmap(self.vis_bgr))

    def on_mouse_release(self, event):
        if self.base_bgr is None: return
        if self.mode == "eraser":
            self.is_painting = False
            self.current_region = None
            self.refresh_view()
            return

        if self.is_painting and self.current_region:
            painted = self.current_region["painted_area"]
            if painted.sum() > 0:
                self.brush_regions.append(self.current_region)
                self.all_painted_mask |= painted
                self.update_region_list_ui()
                self.region_list_widget.setCurrentRow(len(self.brush_regions)-1)
        
        self.current_region = None
        self.is_painting = False
        self.refresh_view()

    def start_new_trajectory(self):
        if self.base_bgr is None: return
        if self.chk_lock_default.isChecked():
            if self.combo_task_type.currentIndex() != self.default_task_type_index:
                self.combo_task_type.setCurrentIndex(self.default_task_type_index)
            else:
                self.on_task_type_changed(self.combo_task_type.currentIndex())
        else:
            self.on_task_type_changed(self.combo_task_type.currentIndex())

        self.brush_regions = []
        self.all_painted_mask = np.zeros((self.H, self.W), dtype=bool)
        self.current_region = None
        self.is_painting = False
        self.selected_region_indices = set()
        self.update_region_list_ui()
        self.refresh_view()

    def on_delete_traj_clicked(self):
        idx = self.combo_traj_select.currentIndex()
        if idx>=0 and idx<len(self.trajectories):
            self.trajectories.pop(idx)
            self.update_traj_view_text()
    
    def on_overwrite_traj_clicked(self):
        idx = self.combo_traj_select.currentIndex()
        if idx>=0 and idx<len(self.trajectories):
            t = self.build_traj()
            if t:
                self.trajectories[idx] = t
                self.update_traj_view_text()
                QMessageBox.information(self,"成功","已重写")
    
    def finish_trajectory(self):
        t = self.build_traj()
        if t:
            self.trajectories.append(t)
            self.update_traj_view_text()
            QMessageBox.information(self,"成功","已保存到列表")

    def build_traj(self):
        if not self.brush_regions: return None
        g, p = [], []
        for r in self.brush_regions:
            obj = {
                "id": int(r["id"]), "mask_index": int(r["mask_index"]),
                "painted_coords": np.argwhere(r["painted_area"]).tolist()
            }
            if r.get("type") == "place": p.append(obj)
            else: g.append(obj)
        
        return {
            "instruction": self.edit_instruction.text(),
            "task_type": TASK_TYPE_MAP.get(self.combo_task_type.currentText(), "unknown"),
            "action": {"actions": self.selected_actions},
            "grasp_regions": g, "place_regions": p
        }

    def save_npz(self):
        if not self.npz_path: return
        self.content["trajectories"] = np.array(json.dumps(self.trajectories), dtype=np.str_)
        np.savez_compressed(self.npz_path, **self.content)
        QMessageBox.information(self,"成功",f"已保存: {os.path.basename(self.npz_path)}")
        
        # 更新 metadata 中的标注状态
        for meta in self.folder_metadata:
            if meta["path"] == self.npz_path:
                meta["annotated"] = True
                break
        
        # 刷新文件名显示
        self.update_file_list_by_filter()

    # 鼠标事件
    def on_mouse_press(self, event):
        self._logic_mouse_press(event)
    def on_mouse_move(self, event, painting=True):
        self._logic_mouse_move(event, painting)

    def _logic_mouse_press(self, event):
        if self.base_bgr is None: return
        x = event.pos().x() - (self.image_label.width() - self.image_label.pixmap().width())//2
        y = event.pos().y() - (self.image_label.height() - self.image_label.pixmap().height())//2
        if x<0 or y<0: return
        
        if self.mode == "eraser":
            self.apply_erase_at(x,y)
            self.refresh_view()
            return
            
        cands = []
        for i,r in enumerate(self.results):
            if point_in_mask(x,y,self.instance_masks[r["mask_index"]]): cands.append((i,r))
        if not cands: return
        
        best = choose_nearest(cands, x, y)
        midx = best[1]["mask_index"]
        
        brush = create_brush_mask(self.H, self.W, x, y, BRUSH_RADIUS)
        mask = (self.instance_masks[midx]>0) & brush & (~self.all_painted_mask)
        
        self.current_region = {
            "id": best[1]["id"], "mask_index": midx, 
            "painted_area": mask, "type": self.mode
        }
        self.current_mask_idx = midx
        self.is_painting = True
        self.refresh_view()

    def _logic_mouse_move(self, event, painting):
        if self.base_bgr is None: return
        x = event.pos().x() - (self.image_label.width() - self.image_label.pixmap().width())//2
        y = event.pos().y() - (self.image_label.height() - self.image_label.pixmap().height())//2
        
        cands = []
        if 0<=x<self.W and 0<=y<self.H:
            for i,r in enumerate(self.results):
                if point_in_mask(x,y,self.instance_masks[r["mask_index"]]): cands.append((i,r))
        self.hover_mask_idx = choose_nearest(cands, x, y)[1]["mask_index"] if cands else None
        
        if painting and self.is_painting and self.current_region:
            if self.mode != "eraser" and self.hover_mask_idx == self.current_mask_idx:
                 brush = create_brush_mask(self.H, self.W, x, y, BRUSH_RADIUS)
                 self.current_region["painted_area"] |= ((self.instance_masks[self.current_mask_idx]>0)&brush)
                 self.refresh_view()
            elif self.mode == "eraser":
                 self.apply_erase_at(x,y)
                 self.refresh_view()
        else:
            self.refresh_view((x,y), self.hover_mask_idx)

    def apply_erase_at(self, x, y):
        if not self.brush_regions: return
        cands = []
        for i,r in enumerate(self.results):
            if point_in_mask(x,y,self.instance_masks[r["mask_index"]]): cands.append((i,r))
        if not cands: return
        best = choose_nearest(cands, x, y)
        midx = best[1]["mask_index"]
        
        brush = create_brush_mask(self.H, self.W, x, y, BRUSH_RADIUS)
        erase_mask = brush & (self.instance_masks[midx]>0)
        
        changed = False
        new_list = []
        for r in self.brush_regions:
            if r["mask_index"] == midx:
                bak = r["painted_area"].copy()
                r["painted_area"][erase_mask] = False
                if not np.array_equal(bak, r["painted_area"]): changed = True
                if r["painted_area"].any(): new_list.append(r)
            else:
                new_list.append(r)
        
        self.brush_regions = new_list
        if changed:
            self.all_painted_mask = np.zeros((self.H, self.W), dtype=bool)
            for r in self.brush_regions: self.all_painted_mask |= r["painted_area"]
            self.update_region_list_ui()

    def on_brush_size_changed(self, val):
        global BRUSH_RADIUS
        BRUSH_RADIUS = max(1, val)
        self.brush_size_label.setText(str(BRUSH_RADIUS))
        self.refresh_view()
        self.update_cursor()
    
    def on_mode_changed(self):
        if self.radio_grasp.isChecked(): self.mode = "grasp"
        elif self.radio_place.isChecked(): self.mode = "place"
        else: self.mode = "eraser"
        self.update_cursor()

    def update_cursor(self):
        if self.base_bgr is None: return
        r = max(3, int(BRUSH_RADIUS))
        size = min(128, r*2+8)
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        c = QColor(0,255,0) if self.mode=="grasp" else (QColor(255,0,0) if self.mode=="place" else QColor(255,255,255))
        painter.setPen(QPen(c, 2))
        painter.drawEllipse(size//2-r, size//2-r, 2*r, 2*r)
        painter.end()
        self.image_label.setCursor(QCursor(pix, size//2, size//2))

    def on_task_type_changed(self, index):
        key = TASK_TYPE_MAP.get(self.combo_task_type.itemText(index))
        self.selected_actions = list(TASK_SPECIFIC_DEFAULTS.get(key, ACTION_POOL))
        self.update_action_sequence_view()
    
    def update_action_sequence_view(self):
        self.actions_view.setText(" -> ".join(self.selected_actions) if self.selected_actions else "(无)")
    
    def on_action_button_clicked(self):
        self.selected_actions.append(self.sender().text())
        self.update_action_sequence_view()
    
    def on_clear_actions_clicked(self):
        self.selected_actions = []
        self.update_action_sequence_view()
    
    def on_default_actions_clicked(self):
        self.on_task_type_changed(self.combo_task_type.currentIndex())
    
    def on_lock_default_toggled(self, checked):
        if checked: self.default_task_type_index = self.combo_task_type.currentIndex()

    def cvimg_to_qpixmap(self, bgr):
        if bgr is None: return QPixmap()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888))

    def on_file_selected(self, idx):
        if idx>=0: self.load_npz_file(self.combo_files.itemData(idx))
    
    def load_npz_file(self, p):
        try:
            d = np.load(p, allow_pickle=True)
            self.npz_path = p
            # self.lbl_file.setText(os.path.basename(p)) # 移出，因为现在文件显示在下拉框里
            self.content = dict(d)
            self.image_rgb = d["image"]
            self.base_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
            self.H, self.W = self.base_bgr.shape[:2]
            self.instance_masks = d["instance_masks"]
            self.results = json.loads(str(d["results"]))
            
            raw = d.get("trajectories", [])
            if isinstance(raw, np.ndarray): raw = raw.item() if raw.ndim==0 else raw.reshape(-1)[0]
            try: self.trajectories = json.loads(raw)
            except: self.trajectories = []
            
            self.start_new_trajectory()
            self.update_traj_view_text()
        except Exception as e:
            print(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec_())