# 自己训练一个视觉语言模型 VLM

搭建一个能「看图回答问题」的视觉语言模型（VLM）。

- 数据：COCO 图片
- 标注：Qwen3.5-9B 自动生成问答
- 模型：SigLIP + Projector + Qwen3-1.7B（VLM v1）
- 流程：环境 → 数据 → vLLM部署 → 生成 QA → 设计 → 训练 → 测试

---

## Step 0：环境搭建


| 组件      | 版本                          |
| ------- | --------------------------- |
| Python  | 3.12                        |
| CUDA    | 13.0（cu130）                 |
| PyTorch | 2.11.0+cu130（由 vLLM 依赖自动安装） |
| vLLM    | ≥ 0.8.5                     |


```bash
conda create -n vlm python=3.12 -y
conda activate vlm
pip config set global.cache-dir $(pwd)/.pip-cache
pip install -r requirements.txt
```

pip 缓存目录：`VLM/.pip-cache/`（数据盘，不占系统盘）

若 `numpy` 下载超时，单独下载后安装再继续：

```bash
mkdir -p wheels
pip download numpy==2.3.5 -d wheels \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --python-version 3.12 --only-binary=:all:
pip install wheels/numpy-*.whl
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

```
2.11.0+cu130
13.0
True
NVIDIA GeForce RTX 5090
```

---



## Step 1：数据准备

下载 [COCO2017 val2017.zip](https://www.modelscope.cn/datasets/PAI/COCO2017/files)：

```bash
mkdir -p data/COCO2017
modelscope download --dataset PAI/COCO2017 val2017.zip --local_dir data/COCO2017
unzip data/COCO2017/val2017.zip -d data/COCO2017
```

解压后 `data/COCO2017/val2017/` 共 **5000** 张图片。

---



## Step 2：vLLM 部署 Qwen3.5-9B



### 2.1 下载模型（第一次）

```bash
mkdir -p models
modelscope download --model Qwen/Qwen3.5-9B --local_dir models/Qwen3.5-9B
modelscope download --model google/siglip2-so400m-patch16-384 --local_dir models/siglip2-so400m-patch16-384
modelscope download --model Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B
```



### 2.2 CUDA 配置（第一次，5090 + FlashInfer）

```bash
conda activate vlm

pip install --force-reinstall --no-deps \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-cccl==13.0.85

export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13
ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
```

JIT 编译报错时执行：`rm -rf ~/.cache/flashinfer`

### 2.3 启动服务（每次）

```bash
conda activate vlm
cd /root/autodl-fs/VLM

export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH

vllm serve models/Qwen3.5-9B --port 8033 --reasoning-parser qwen3
```



### 2.4 测试

curl 中 `model` 用本地路径 `models/Qwen3.5-9B`（与 `vllm serve` 一致）。默认关闭推理：`chat_template_kwargs.enable_thinking: false`。

```bash
curl http://localhost:8033/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B",
    "messages": [{"role":"user","content":"你好"}],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 2048,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

测试脚本（`models/demo/`）：

```bash
cd /root/autodl-fs/VLM
python models/demo/qwen3_5_9b.py   # 需先启动 2.3 服务
python models/demo/qwen3_1_7b.py   # vLLM 占用 GPU 时需先停服务
python models/demo/siglip2.py
```

---



## Step 3：生成视觉问答数据

用 Qwen3.5-9B（vLLM）对 COCO 图片自动生成问答，供后续训练 VLM 使用。

### 3.1 问题模板

`utils/question_templates.py` 定义 **6 类问题**，每类 **5 种问法**，生成时每张图每类 **随机选 1 种**：


| category | category_name | 说明       |
| -------- | ------------- | -------- |
| 1        | 简要描述          | 图片主要内容   |
| 2        | 主要对象          | 场景中的主要物体 |
| 3        | 对象颜色          | 主要物体颜色   |
| 4        | 对象动作          | 主要物体在做什么 |
| 5        | 室内室外          | 室内 / 室外  |
| 6        | 天气情况          | 图中天气     |


每张图 → **6 条**问答；5000 张图 → **30000 条**。

### 3.2 生成（需先启动 Step 2.3 服务）

```bash
conda activate vlm
cd /root/autodl-fs/VLM/utils

# 测试 10 张
python generate_qa.py --num-images 10

# 全部 5000 张
python generate_qa.py --num-images 0
```

默认输出：`data/qa/coco_val_qa.json`。每处理 **50 张**图片写入一次（`--batch-size` 可调）。**10 路并发**请求（`--workers 10`）。

常用参数：

```bash
python generate_qa.py \
  --num-images 0 \
  --output ../data/qa/coco_val_qa.json \
  --batch-size 50 \
  --workers 10 \
  --seed 42          # 固定随机问法，便于复现
```



### 3.3 输出格式

每条 JSON 记录：

```json
{
  "image": "data/COCO2017/val2017/000000000139.jpg",
  "category": 1,
  "category_name": "简要描述",
  "question": "这张图片的核心内容是什么？请简要说明。",
  "answer": "..."
}
```

生成时使用 system prompt 约束模型 **简要回答、不过度解释**，并默认关闭推理模式。

---



## Step 4：VLM v1 模型架构

**目标**：让 Qwen3-1.7B 能「看图说话」——把 SigLIP 提取的视觉特征对齐到 LLM 的语义空间，再按问答格式生成回答。

**整体流水线**（Step 0 → 6）：

```
环境 → COCO 图片 → vLLM 部署 Qwen3.5-9B → 自动生成 QA → 设计 VLM v1 → 训练 → 测试
```

**三模块**（代码：`models/vlms/VLM_v1_model.py`）：


| 模块        | 模型             | 训练     | 作用                                        |
| --------- | -------------- | ------ | ----------------------------------------- |
| Vision    | SigLIP2-so400m | 冻结     | 384×384 图片 → 576 个 patch 特征 `(576, 1152)` |
| Projector | 2 层 MLP + GELU | **训练** | `1152 → 2048`，对齐 Qwen 隐层维度                |
| LLM       | Qwen3-1.7B     | 冻结     | 读 multimodal 序列，自回归生成 answer              |


**图文怎么拼**：prompt 里放 1 个 `<image>` 占位符；forward 时将其展开为 **576 个视觉 token**，与文本 embedding 沿序列维拼接，再送入 Qwen。

举例（假设 token 数）：

```
question = "这张图里有什么？"
answer   = "图片中有一只猫和一张沙发。"

text embed (28, 2048)
  → [prefix 3] + [576 视觉 token] + [suffix 24]
  → merged (603, 2048)  →  Qwen3  →  自回归生成 answer
```

验证架构（需 GPU，vLLM 占用时需先停服务）：

```bash
cd /root/autodl-fs/VLM
python models/vlms/VLM_v1_model.py
```

---



## Step 5：训练 VLM v1

**训练策略**：

- 数据：Step 3 生成的 `coco_val_qa.json`（5000 图 × 6 问 ≈ 30000 条）
- 方式：SFT（instruction tuning），prompt 部分 labels 置 `-100`，只对 answer 算 loss
- 序列上限：`max_seq_len = 704`（576 图 + 128 文本），answer 超长则截断
- 只更新 Projector；Vision / LLM 权重不动

默认只训练 **Projector**：

```bash
conda activate vlm
cd /root/autodl-fs/VLM

# 小规模试跑
python scripts/train_vlm_v1.py --max-samples 100 --epochs 1

# 全量
python scripts/train_vlm_v1.py --epochs 1 --batch-size 1 --grad-accum 4
```

常用参数：


| 参数              | 默认                   | 说明                                  |
| --------------- | -------------------- | ----------------------------------- |
| `--max-seq-len` | 704                  | 合并后最大序列长度（576 图 + 128 文本）           |
| `--save-every`  | 1000                 | 每 N 次 optimizer step 存一次；`0` 仅结束时保存 |
| `--output`      | `checkpoints/VLM_v1` | 输出目录                                |


- 中间 checkpoint：`projector_step_1000.pt` …
- 最终权重：`checkpoints/VLM_v1/projector.pt`

序列长度分布可用 `python utils/analyze_seq_len.py` 查看。

---



## Step 6：测试 VLM v1

训练完成后，用 `scripts/test_vlm_v1.py` 对固定图片列表做推理：每张图依次问 **6 类问题的第一种问法**（与 Step 3 问题模板一致）。

默认测试列表（脚本内 `TEST_IMAGES`，可按需修改）：


| 图片                                       |
| ---------------------------------------- |
| `data/COCO2017/val2017/000000000139.jpg` |
| `data/COCO2017/val2017/000000000285.jpg` |
| `data/COCO2017/val2017/000000000632.jpg` |


```bash
conda activate vlm
cd /root/autodl-fs/VLM

# 默认：TEST_IMAGES 全部图片 + 6 类问题
python scripts/test_vlm_v1.py

# 指定 checkpoint
python scripts/test_vlm_v1.py --checkpoint checkpoints/VLM_v1/projector_step_1000.pt
```

常用参数：


| 参数                 | 默认                                | 说明                        |
| ------------------ | --------------------------------- | ------------------------- |
| `--checkpoint`     | `checkpoints/VLM_v1/projector.pt` | projector 权重；`none` 表示不加载 |
| `--max-new-tokens` | 128                               | 生成长度上限                    |


需 GPU；vLLM 占用显存时需先停 Step 2.3 服务。