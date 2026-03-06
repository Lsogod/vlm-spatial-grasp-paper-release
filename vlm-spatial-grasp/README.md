# VLM_Grasp_Interactive

本项目对应论文《基于视觉提示的具身智能空间感知增强方法研究》，实现了一个面向抓取任务的训练无关（training-free）空间感知增强闭环：  
`结构化视觉锚点（SVA） -> 结构化视觉提示（SVP） -> 像素落点回写 -> 抓取姿态推理 -> 机械臂执行`。

## 1. 论文方法与代码对应

### 1.1 SVA（结构化视觉锚点）
- 目标：把“连续坐标生成”转换为“可索引的离散候选引用”。
- 代码入口：`vision_agent_v2.py`
- 实现路径：
  - 调用后端 `POST /vision/sam` 获取对象编号可视化与 `npz_path`
  - 调用后端 `POST /vision/dino` 获取目标编号的关键点（锚点）集合
  - 形成 `id -> pixel` 映射表，作为后续动作参数的空间锚定依据

### 1.2 SVP（结构化视觉提示）
- 目标：通过三阶段提示词，将开放式输出约束为结构化决策。
- 代码入口：`vision_agent_v2.py::plan_and_ground_robot_actions`
- 三阶段推理：
  - Stage1：意图解析 + 目标对象/部位编号提取
  - Stage2：动作序列规划（`action_components`）
  - Stage3：对象部位到编号映射（`id`）
- 最终把编号回写为像素：`action_components[*].args.pixel = [u,v]`

### 1.3 闭环执行（感知到控制）
- 入口：`main_vlm.py`
- 核心流程：
  1. MuJoCo 环境渲染 RGB-D（`UR5GraspEnv`）
  2. `plan_and_ground_robot_actions` 生成结构化动作与像素落点
  3. `run_grasp_inference` 在上下文掩码中做抓取候选生成
  4. 约束半径内筛选抓取、碰撞过滤、角度过滤、评分排序
  5. `GraspFlow` 执行动作（抓取/移动/开合夹爪/归零）

### 1.4 Qwen 实现（仓库内已有，默认未启用）
- 视觉 grounding（目标/部位定位）Qwen 入口：`vision_agent.py::call_model_api`
- 相关环境变量：`MODELSCOPE_BASE_URL`、`MODELSCOPE_API_KEY`、`MODELSCOPE_MODEL_ID`
- 说明：当前默认主线是 `vision_agent_v2.py`（Azure + 后端 SAM/DINO）。Qwen 路径作为可切换实现保留在仓库中。

## 2. 目录结构（核心）

```text
VLM_Grasp_Interactive/
|- main_vlm.py                # 端到端闭环主入口（推荐）
|- vision_agent_v2.py         # SVA+SVP 三阶段推理与像素回写
|- vision_agent.py            # 可切换的 Qwen grounding 版本
|- grasp_process.py           # GraspNet 推理与抓取过滤
|- vlm_process.py             # SAM 相关处理与旧版流程函数
|- show_depth_live.py         # 深度图实时可视化
|- sam_b.pt                   # SAM 权重
|- logs/log_rs/checkpoint-rs.tar  # GraspNet 权重
|- graspnet-baseline/         # GraspNet 基线与算子
|- manipulator_grasp/         # UR5e+夹爪+MuJoCo 环境
```

## 3. 环境准备

推荐环境：`Python 3.10`、`CUDA + PyTorch`（CPU 也可运行但较慢）。

### 3.1 安装基础依赖

```bash
pip install numpy scipy pillow tqdm opencv-python matplotlib requests
pip install torch open3d ultralytics openai python-dotenv
pip install mujoco glfw spatialmath-python roboticstoolbox-python modern-robotics pandas
```

### 3.2 安装 GraspNet 相关扩展

```bash
cd graspnet-baseline/pointnet2
python setup.py install

cd ../knn
python setup.py install

cd ../graspnetAPI
pip install .
```

## 4. 模型与资源检查

运行前请确认以下文件存在：

- `sam_b.pt`
- `logs/log_rs/checkpoint-rs.tar`

缺失时会导致分割或抓取推理失败。

## 5. 环境变量（建议）

### 5.0 方法切换（`main_vlm.py`）

- `VLM_METHOD`：`svp`（默认，论文主方法）或 `qwen`（Qwen grounding）

### 5.1 默认主线（`vision_agent_v2.py`）

- `BACKEND_URL`（示例 `http://<your-backend-host>`）
- `AZURE_ENDPOINT`
- `AZURE_DEPLOYMENT`
- `AZURE_API_VERSION`
- `AZURE_KEY`

```bash
BACKEND_URL=http://<your-backend-host>
AZURE_ENDPOINT=https://<your-endpoint>.openai.azure.com
AZURE_DEPLOYMENT=<your-deployment>
AZURE_API_VERSION=2024-02-15-preview
AZURE_KEY=<your-key>
```

### 5.2 Qwen 路径（`vision_agent.py` 中的 grounding）

- `MODELSCOPE_BASE_URL`（默认 `https://api-inference.modelscope.cn/v1`）
- `MODELSCOPE_API_KEY`
- `MODELSCOPE_MODEL_ID`（默认 `qwen3-vl-plus-2025-12-19`）

```bash
MODELSCOPE_BASE_URL=https://api-inference.modelscope.cn/v1
MODELSCOPE_API_KEY=<your-modelscope-key>
MODELSCOPE_MODEL_ID=qwen3-vl-plus-2025-12-19
```

说明：`vision_agent.py` 的动作规划步骤仍调用 Azure（变量名为 `AZURE_OPENAI_*`），Qwen 主要用于视觉定位（grounding）。

请勿把真实密钥提交到仓库。

## 6. 快速开始

### 6.1 可选：先看深度流

```bash
python show_depth_live.py
```

### 6.2 运行端到端闭环

```bash
python main_vlm.py
```

程序启动后输入自然语言指令，例如：

- `将小黄鸭放在魔方上面`
- `抓住香蕉中部并抬起`

### 6.3 单独测试三阶段推理（不执行机械臂）

```bash
python vision_agent_v2.py --img images/your_image.jpg --ins "将香蕉放到杯子旁边" --save logs/run1 --log logs/run1.jsonl
```

### 6.4 切换到 Qwen 实现（可选）

`main_vlm.py` 已支持环境变量切换，无需手改 import：

```powershell
# 默认：论文主方法链路（SVP）
$env:VLM_METHOD="svp"
python main_vlm.py

# 切到 Qwen grounding（vision_agent）
$env:VLM_METHOD="qwen"
python main_vlm.py
```

```bash
# Linux / macOS
export VLM_METHOD=svp
python main_vlm.py

export VLM_METHOD=qwen
python main_vlm.py
```

推荐值：`svp` / `qwen`。  
兼容旧变量：`VLM_AGENT_IMPL`（未设置 `VLM_METHOD` 时仍可用）。

## 7. 关键输出

- `logs/run1.jsonl`：三阶段推理与后端交互日志
- `logs/run1_dino_vis.jpg`：编号/关键点可视化
- `action_json_str`：结构化动作序列（含 `args.pixel`）

## 8. 关键参数

`main_vlm.py` 中常用配置：

- `CONTEXT_RADIUS_PX = 80`：提供给抓取推理的上下文范围
- `CONSTRAINT_RADIUS_PX = 30`：最终抓取候选的像素约束半径

`vision_agent_v2.py` 中常用配置：

- `min_area`：SAM 最小区域过滤
- `azure_timeout` / `sam_timeout` / `dino_timeout`：推理超时

## 9. 结果

在统一静态评估协议下（无需额外坐标标注和梯度训练），论文报告相较坐标监督回归基线：

- 抓取命中率提升约 `+9.17` 个百分点
- 语义-空间一致性提升约 `+9.10` 个百分点
- 抓取偏差降低约 `51.3%`

本仓库即面向该“离散候选+结构化提示+执行前校验”的工程实现。

## 10. 常见问题

- `Backend /vision/sam Error` 或 `Backend /vision/dino Error`  
  检查 `BACKEND_URL` 可达性与接口返回格式。

- `AZURE_KEY is empty` 或 Azure 请求失败  
  检查 Azure 端点、部署名、API 版本、密钥是否匹配。

- Qwen 请求失败或 401  
  检查 `MODELSCOPE_API_KEY`、`MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID` 是否正确。

- `No grasp predictions` 或约束后为空  
  适当增大 `CONSTRAINT_RADIUS_PX`，或先确认目标像素是否位于物体实体区域。

- MuJoCo 可视化异常  
  检查图形驱动、`mujoco/glfw` 版本，必要时先用 `show_depth_live.py` 做环境验证。
