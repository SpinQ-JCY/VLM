# VLM v1：自己训练一个视觉语言模型

SigLIP2（视觉）+ Projector（对齐层）+ Qwen3-1.7B（语言），**只训练 Projector**。

两阶段：**align** 用 COCO-CN 中文描述做语义预热 → **sft** 用 Qwen3.5 生成的大规模问答做指令微调。**推理与 Web 演示请用 sft 权重**；align 数据少、问题单一，单独使用效果较差，主要供 sft 初始化。

---

## 整体流程（Step 0 → 7）

```
Step 0  环境          conda 环境 vlm + requirements.txt
Step 1  数据          COCO2014 图片 + COCO-CN 标注  →  data/COCO2014/
Step 2  vLLM          Qwen3.5-9B 服务                →  localhost:8033
Step 3  训练数据      3.1 align 20,341 条 | 3.2 sft 165,566 条
Step 4  模型架构      models/vlms/VLM_v1_model.py
Step 5  训练          align → sft（仅更新 Projector）
Step 6  命令行测试    scripts/step6_test_vlm_v1.py
Step 7  Web 界面      web/server.py :7860
```

---

## 仓库结构（推送内容）

```
VLM/
├── models/vlms/VLM_v1_model.py   # 模型定义
├── models/demo/                  # 各组件 Demo
├── utils/
│   ├── step3_build_coco_cn_qa.py # Step 3.1
│   ├── step3_generate_qa.py      # Step 3.2
│   └── step5_train_logger.py
├── scripts/
│   ├── step5_train_vlm_v1.py     # Step 5
│   └── step6_test_vlm_v1.py      # Step 6
├── web/                          # Step 7
├── requirements.txt
└── README.md
```

不推送（见 `.gitignore`）：`models/*` 大模型权重、`data/`、`logs/`、中间 checkpoint（`projector_step_*.pt`）。

**已随仓库提供**（clone 可直接推理）：

| 文件 | 说明 |
|------|------|
| `checkpoints/VLM_v1_align/projector.pt` | align 阶段最终权重 |
| `checkpoints/VLM_v1_sft/projector.pt` | sft 阶段最终权重（**推荐推理**） |

QA 数据格式（两阶段相同）：

```json
{"image": "data/COCO2014/.../xxx.jpg", "question": "...", "answer": "..."}
```

---

## Step 0：环境

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| CUDA | 13.0（cu130） |
| PyTorch | 2.11.0+cu130 |
| vLLM | ≥ 0.8.5 |

```bash
conda create -n vlm python=3.12 -y && conda activate vlm
cd VLM
pip config set global.cache-dir $(pwd)/.pip-cache
pip install -r requirements.txt
```

---

## Step 1：数据

**1.1 COCO2014 图片** — [ModelScope 下载](https://www.modelscope.cn/datasets/OmniData/COCO_2014/tree/master/raw)

```bash
mkdir -p data/COCO2014/raw
modelscope download --dataset OmniData/COCO_2014 raw/train2014.zip --local_dir data/COCO2014
modelscope download --dataset OmniData/COCO_2014 raw/val2014.zip --local_dir data/COCO2014
unzip data/COCO2014/raw/train2014.zip -d data/COCO2014
unzip data/COCO2014/raw/val2014.zip -d data/COCO2014
```

本地规模：`train2014` **82,783** 张 · `val2014` **40,504** 张。

**1.2 COCO-CN 标注** — [HuggingFace](https://huggingface.co/datasets/AIMClab-RUC/COCO-CN)（慢可用 `HF_ENDPOINT=https://hf-mirror.com`）

```bash
wget -c -O data/COCO2014/coco-cn-version1805v1.1.tar.gz \
  "https://huggingface.co/datasets/AIMClab-RUC/COCO-CN/resolve/main/coco-cn-version1805v1.1.tar.gz?download=true"
tar -xzf data/COCO2014/coco-cn-version1805v1.1.tar.gz -C data/COCO2014
```

---

## Step 2：vLLM（生成 sft 数据用）

**下载模型（首次）**

```bash
mkdir -p models
modelscope download --model Qwen/Qwen3.5-9B --local_dir models/Qwen3.5-9B
modelscope download --model google/siglip2-so400m-patch16-384 --local_dir models/siglip2-so400m-patch16-384
modelscope download --model Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B
```

**5090 + FlashInfer（首次）**：安装 cu13 工具链并设 `CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13`；JIT 报错时 `rm -rf ~/.cache/flashinfer`。

**启动（每次 Step 3.2 前）**

```bash
conda activate vlm && cd VLM
export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
vllm serve models/Qwen3.5-9B --port 8033 --reasoning-parser qwen3
```

---

## Step 3：构建训练数据

### 3.1 align — `utils/step3_build_coco_cn_qa.py`

COCO-CN `#0` caption + 固定问题「请简要描述图片主要内容」→ `data/qa/coco_cn_qa.json`

```bash
python utils/step3_build_coco_cn_qa.py
```

### 3.2 sft — `utils/step3_generate_qa.py`

Qwen3.5 **看图**生成：每图 **2 问**（10 种描述问法随机 1 + 自拟细节问），答 ≤30/15 字 → `data/qa/coco_train_qa_qwen3.5.json`。支持断点续跑。

```bash
python utils/step3_generate_qa.py --num-images 10   # 试跑
python utils/step3_generate_qa.py --num-images 0    # 全量 train2014
```

生成完成后**停 vLLM**，再训练。

### 数据对比

| | align | sft |
|---|-------|-----|
| 文件 | `coco_cn_qa.json` | `coco_train_qa_qwen3.5.json` |
| 条数 | 20,341 | 165,566 |
| 图片 | 20,341（COCO-CN 子集） | 82,783（train 全量） |
| 问答 | 1 固定问 + 人工 caption | 2 问/图，Qwen3.5 生成 |

---

## Step 4：架构

| 模块 | 模型 | 训练 |
|------|------|------|
| Vision | SigLIP2-so400m | 冻结 |
| Projector | 2×MLP | **训练** |
| LLM | Qwen3-1.7B | 冻结 |

`<image>` 占位符展开为 576 视觉 token，与文本拼接后送入 Qwen。验证：`python models/vlms/VLM_v1_model.py`

---

## Step 5：训练 — `scripts/step5_train_vlm_v1.py`

prompt labels = `-100`，只对 answer 算 loss；`max_seq_len=704`（576 图 + 128 文本）。

```bash
conda activate vlm && cd VLM

# align
python scripts/step5_train_vlm_v1.py --phase align

# sft（默认加载 align 的 projector.pt）
python scripts/step5_train_vlm_v1.py --phase sft
```

| 阶段 | 数据 | 输出 |
|------|------|------|
| align | `data/qa/coco_cn_qa.json` | `checkpoints/VLM_v1_align/projector.pt` |
| sft | `data/qa/coco_train_qa_qwen3.5.json` | `checkpoints/VLM_v1_sft/projector.pt` |

日志：`logs/VLM_v1_<phase>/train.log`、`loss.png`。全量 sft 可后台：

```bash
mkdir -p logs/VLM_v1_sft
nohup python scripts/step5_train_vlm_v1.py --phase sft > logs/VLM_v1_sft/nohup.out 2>&1 &
```

---

## Step 6：命令行测试 — `scripts/step6_test_vlm_v1.py`

```bash
python scripts/step6_test_vlm_v1.py                                          # 默认 sft
python scripts/step6_test_vlm_v1.py --checkpoint checkpoints/VLM_v1_align/projector.pt  # 对比 align
```

---

## Step 7：Web 界面 — `web/server.py`

```bash
conda activate vlm && cd VLM/web
python server.py
```

默认 `checkpoints/VLM_v1_sft/projector.pt`，端口 **7860**。上传图片后逐条提问，无多轮上下文。

**访问方式**

| 环境 | 做法 |
|------|------|
| 本机 / 公司局域网 | 终端打印 `192.168.x.x:7860`，发给同网同事 |
| AutoDL（Docker） | `172.17.x.x` 是容器 IP，**不能**当局域网用；在控制台做 **7860 端口映射**，用平台外链分享 |

---

> 以下命令若无特别说明，均在项目根目录 `VLM/` 下、已 `conda activate vlm` 后执行。
