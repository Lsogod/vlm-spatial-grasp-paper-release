# Paper Release Bundle

## 1. 数据是否与论文对应

对应。当前 release 中论文主实验统一使用同一批评测样本：

- 主评测数据：`data/dist_all`（160 个 `.npz`）
- 三域划分规则（按文件名前缀数字）：
  - `0-99` -> `GraspNet-Desk`
  - `100-129` -> `Real-Desk`
  - `130-159` -> `Sim-Desk`

该划分与统计口径用于论文中的三域对比与全量评测。

## 2. 论文主结果相关目录

```text
paper_release_bundle/
|- data/
|  |- dist_all/                          # 论文主评测数据（160 个样本）
|- experiments/
|  |- table3_3.py                        # 生成“表3-4对应”的三域静态评估（含柱状图）
|  |- table4_1.py                        # 生成表4-1（Full, Oracle-Action）
|  |- table4_2.py                        # 生成表4-2（Strict, Supplementary）
|- results/
   |- predictions_json/                  # 各方法预测 json
   |- tables/                            # 各表格 CSV 输出
```

## 3. 论文表格与代码/数据一一对应

| 论文表号 | 表格含义 | 输入数据 | 方法结果输入 | 生成脚本 | 对应输出文件 |
|---|---|---|---|---|---|
| 表3-2 | 三域数据规模统计（数据基准划分） | `data/dist_all` | 无 | 口径由脚本中的域划分规则定义 | 由 `dist_all` 文件名区间统计得到 |
| 表3-4 | 三域静态评估（锚点/落点策略对比） | `data/dist_all` | `results/predictions_json/ai_results` 或 `ai_results_4o` | `experiments/table3_3.py` | `results/tables/table3_3_ai_results/summary.csv`、`results/tables/table3_3_ai_results_4o/summary.csv` |
| 表4-1 | 全量评测（Full）静态指标对比，Oracle-Action | `data/dist_all` | `results/predictions_json/*_results` | `experiments/table4_1.py` | `results/tables/table4_full.csv`、`results/tables/table4_full_recomputed.csv` |
| 表4-2 | 全量评测（Full）严格一致性评估（Supplementary） | `data/dist_all` | `results/predictions_json/*_results` | `experiments/table4_2.py` | `results/tables/table4_2_strict.csv`、`results/tables/table4_2_strict_recomputed.csv` |

## 4. 每张论文表的详细含义

### 表3-2（三域数据规模）

- 用途：定义数据基准与分域统计口径。
- 含义：不是“模型输出结果表”，而是“评测数据如何划分与计数”。
- 来源：`data/dist_all` 的样本命名前缀区间（`0-99/100-129/130-159`）。

### 表3-4（三域静态评估）

- 用途：比较落点策略在三个域上的静态空间表现。
- 指标（summary.csv 中）：
  - `A_hit_g`：抓取阶段命中率（越高越好）
  - `D_spatial_g`：抓取阶段空间偏差（像素，越低越好）
  - `A_hit_p`：放置阶段命中率（越高越好）
  - `D_spatial_p`：放置阶段空间偏差（像素，越低越好）
  - `A_joint`：联合一致性指标（越高越好）
- 输出：`summary.csv` + 5 张柱状图（`bar_*.png`）。

### 表4-1（Full, Oracle-Action）

- 用途：论文主对比表，比较不同方法在全量数据上的静态空间性能。
- 指标：
  - `A_hit_g`, `D_spatial_g_px`
  - `A_hit_p`, `D_spatial_p_px`
  - `A_joint`
- 说明：同一套 `dist_all` + 各方法预测 JSON，统一口径计算。

### 表4-2（Strict, Supplementary）

- 用途：严格一致性补充评估，用于验证语义/轨迹/坐标联合一致性。
- 指标：
  - `InstrMatch`：指令匹配率
  - `TrajSucc`：轨迹级成功率
  - `ActAcc`：动作准确率
  - `CoordHit`：坐标命中率
  - `JointSucc`：联合成功率
  - `AvgDev_px`：平均偏差（像素）


## 5. 可视化结果图（柱状图）

默认展示 `ai_results_4o` 对应的三域可视化：

![A_hit_g](results/tables/table3_3_ai_results_4o/bar_A_hit_g.png)
![A_hit_p](results/tables/table3_3_ai_results_4o/bar_A_hit_p.png)
![A_joint](results/tables/table3_3_ai_results_4o/bar_A_joint.png)
![D_spatial_g](results/tables/table3_3_ai_results_4o/bar_D_spatial_g.png)
![D_spatial_p](results/tables/table3_3_ai_results_4o/bar_D_spatial_p.png)
