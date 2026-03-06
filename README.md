# 基于视觉提示的具身智能空间感知增强方法研究

本仓库用于对齐论文中的四类内容：`数据`、`方法`、`实验`、`结果`。

- 方法工程：`vlm-spatial-grasp`
- 论文复现与结果：`paper_release`

## 1. 数据内容（对应论文数据部分）

论文主评测数据位于：

- `paper_release/data/dist_all`（160 个 `.npz` 样本）

三域划分规则（与论文统计口径一致）：

- `0-99` -> GraspNet-Desk
- `100-129` -> Real-Desk
- `130-159` -> Sim-Desk

每个样本的核心字段（160/160 全覆盖）：

- `image`：RGB 图像
- `labeled_image`：可视化标注图
- `instance_masks`：实例掩码集合
- `results`：实例信息（`id/mask_index/bbox/centroid/area`）
- `trajectories`：任务轨迹与金标准动作（JSON 字符串）

补充字段（仅少量样本存在）：

- `instruction`、`action`
- `keypoints`、`keypoint_ids`、`keypoint_mask_indices`

数据可视化样例：

- `paper_release/data/dataset_vis/0-99_grid.png`
- `paper_release/data/dataset_vis/100-129_grid.png`
- `paper_release/data/dataset_vis/130-159_grid.png`
- `paper_release/data/dataset_vis/summary.txt`

## 2. 方法内容

论文主方法（SVA + SVP + 闭环执行）在 `vlm-spatial-grasp`：

- `vision_agent_v2.py`：结构化视觉锚点（SVA）与结构化视觉提示（SVP）主链路
- `main_vlm.py`：端到端执行入口（支持 `VLM_METHOD=svp/qwen` 切换）
- `grasp_process.py`：抓取候选生成与筛选
- `manipulator_grasp/`：MuJoCo 机械臂执行环境

论文对比方法（基线）在 `paper_release/experiments`：

- `qwen_batch_inference.py`：Qwen 基线
- `gpt_batch_inference.py`：GPT 基线
- `ours_svp_batch_inference.py`：本文方法批量推理

## 3. 实验内容

### 3.1 输入与预测结果目录

- 数据输入：`paper_release/data/dist_all`
- 各方法预测：`paper_release/results/predictions_json/`
  - `ai_results`、`ai_results_4o`
  - `gpt-4o_results`、`gpt-5_results`
  - `qwen_plus_results_newGT`、`qwen_flash_results_newGT`

### 3.2 论文表格对应关系

| 论文表号 | 表格含义 | 实验脚本 | 输出文件 |
|---|---|---|---|
| 表3-2 | 三域数据规模统计 | 按 `dist_all` 文件名前缀区间统计 | 数据划分统计口径 |
| 表3-4 | 三域静态评估（命中率/偏差） | `paper_release/experiments/table3_3.py` | `paper_release/results/tables/table3_3_ai_results*/summary.csv` |
| 表4-1 | Full 评测（Oracle-Action） | `paper_release/experiments/table4_1.py` | `paper_release/results/tables/table4_full.csv`、`table4_full_recomputed.csv` |
| 表4-2 | Strict 补充评测（Supplementary） | `paper_release/experiments/table4_2.py` | `paper_release/results/tables/table4_2_strict.csv`、`table4_2_strict_recomputed.csv` |

说明：脚本名 `table3_3.py` 是历史命名，当前用于生成论文中的表3-4对应结果。

## 4. 结果内容（对应论文结果部分）

论文主结果表（CSV）：

- `paper_release/results/tables/table4_full.csv`
- `paper_release/results/tables/table4_full_recomputed.csv`
- `paper_release/results/tables/table4_2_strict.csv`
- `paper_release/results/tables/table4_2_strict_recomputed.csv`

三域评估结果（表3-4对应）：

- `paper_release/results/tables/table3_3_ai_results/summary.csv`
- `paper_release/results/tables/table3_3_ai_results_4o/summary.csv`

三域柱状图：

- `paper_release/results/tables/table3_3_ai_results_4o/bar_A_hit_g.png`
- `paper_release/results/tables/table3_3_ai_results_4o/bar_A_hit_p.png`
- `paper_release/results/tables/table3_3_ai_results_4o/bar_A_joint.png`
- `paper_release/results/tables/table3_3_ai_results_4o/bar_D_spatial_g.png`
- `paper_release/results/tables/table3_3_ai_results_4o/bar_D_spatial_p.png`

## 5. 非论文主表文件（保留用于调试）

以下文件不作为论文主表对应项：

- `paper_release/results/tables/table4_full_diag.csv`
- `paper_release/results/tables/table4_full_recomputed_diag.csv`
- `paper_release/results/tables/table3_4_*.csv`


