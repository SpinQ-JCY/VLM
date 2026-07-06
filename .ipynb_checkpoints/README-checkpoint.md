# VLM v1：自己训练一个视觉语言模型

SigLIP2（视觉）+ Projector（对齐层）+ Qwen3-1.7B（语言），**只训练 Projector**。

两阶段：**align** 用 COCO-CN 中文描述做语义预热 → **sft** 用 Qwen3.5 生成的大规模问答做指令微调。**推理与 Web 演示请用 sft 权重**；align 数据少、问题单一，单独使用效果较差，主要供 sft 初始化。

---

## 整体流程（Step 0 → 7）

```
Step 0  环境          conda 环境 vlm + requirements.txt
Step 1  数据          COCO2014 图片 + COCO-CN 标注  →  data/COCO2014/
Step 2  vLLM与模型准备  下载权重 + 启动 Qwen3.5-9B  →  localhost:8033
Step 3  训练数据      仓库已带 QA；可选重跑 3.1/3.2 覆盖 data/qa/
Step 4  模型架构      models/vlms/VLM_v1_model.py
Step 5  训练          align → sft（仅更新 Projector）
Step 6  命令行测试    scripts/step6_test_vlm_v1.py
Step 7  Web 界面      web/server.py
```

---



## 仓库结构

```
VLM/
├── data/qa/                   # 训练问答 JSON
├── models/vlms/               # VLM 模型定义（Step 4）
├── models/demo/               # 各组件单独验证（Step 2）
├── utils/                     # 数据构建、训练日志（Step 3、5）
├── scripts/                   # 训练与测试（Step 5、6）
├── web/                       # Web 界面（Step 7）
├── checkpoints/               # align / sft 的 projector.pt
└── requirements.txt
```

---



## Step 0：环境


| 组件      | 版本           |
| ------- | ------------ |
| Python  | 3.12         |
| CUDA    | 13.0（cu130）  |
| PyTorch | 2.11.0+cu130 |
| vLLM    | ≥ 0.8.5      |


```bash
conda create -n vlm python=3.12 -y && conda activate vlm
cd VLM
pip config set global.cache-dir $(pwd)/.pip-cache
pip install -r requirements.txt
```

pip 缓存目录：`VLM/.pip-cache/`（数据盘，不占系统盘）

若某包下载慢或超时，可先单独下载 wheel 到 `wheels/` 再安装，例如 `numpy`：

```bash
mkdir -p wheels
pip download numpy==2.3.5 -d wheels \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --python-version 3.12 --only-binary=:all:
pip install wheels/numpy-*.whl
pip install -r requirements.txt
```

**验证环境**

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期类似（GPU 型号因机器而异）：

```
2.11.0+cu130
13.0
True
NVIDIA GeForce RTX 5090
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

**1.2 COCO-CN 标注** — [HuggingFace](https://huggingface.co/datasets/AIMClab-RUC/COCO-CN)

```bash
wget -c -O data/COCO2014/coco-cn-version1805v1.1.tar.gz \
  "https://huggingface.co/datasets/AIMClab-RUC/COCO-CN/resolve/main/coco-cn-version1805v1.1.tar.gz?download=true"
tar -xzf data/COCO2014/coco-cn-version1805v1.1.tar.gz -C data/COCO2014
```

若 HuggingFace 下载慢，可改用 [hf-mirror](https://hf-mirror.com)：

```bash
wget -c -O data/COCO2014/coco-cn-version1805v1.1.tar.gz \
  "https://hf-mirror.com/datasets/AIMClab-RUC/COCO-CN/resolve/main/coco-cn-version1805v1.1.tar.gz?download=true"
tar -xzf data/COCO2014/coco-cn-version1805v1.1.tar.gz -C data/COCO2014
```

---



## Step 2：vLLM与模型准备

**下载模型（首次）**

```bash
mkdir -p models
modelscope download --model Qwen/Qwen3.5-9B --local_dir models/Qwen3.5-9B
modelscope download --model google/siglip2-so400m-patch16-384 --local_dir models/siglip2-so400m-patch16-384
modelscope download --model Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B
```

**5090 + FlashInfer（首次）**

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

JIT 编译报错时：`rm -rf ~/.cache/flashinfer`

**启动（每次 Step 3.2 前）**

```bash
conda activate vlm && cd VLM

export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH

vllm serve models/Qwen3.5-9B --port 8033 --reasoning-parser qwen3
```

**测试**

curl 中 `model` 用本地路径 `models/Qwen3.5-9B`（与 `vllm serve` 一致）。默认关闭推理：`chat_template_kwargs.enable_thinking: false`。

采样参数：`temperature` 越高越随机、越低越稳；`top_p` 从累积概率达 p 的候选词中采样；`top_k` 只保留概率最高的 k 个候选词。

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
cd VLM
python models/demo/qwen3_5_9b.py   # 需先启动 vLLM 服务
python models/demo/qwen3_1_7b.py   # vLLM 占用 GPU 时需先停服务
python models/demo/siglip2.py
```

---



## Step 3：构建训练数据（可选）

仓库已附带 `data/qa/` 下两份 JSON，**可直接 Step 5 训练**。仅当需要重新生成或修改问法时再跑本节。

### 3.1 align — `utils/step3_build_coco_cn_qa.py`

COCO-CN `#0` caption + 固定问题「请简要描述图片主要内容」→ `data/qa/coco_cn_qa.json`

```bash
python utils/step3_build_coco_cn_qa.py
```

单条样本示例：

```json
{
  "image": "data/COCO2014/train2014/COCO_train2014_000000000036.jpg",
  "question": "请简要描述图片主要内容",
  "answer": "一个年轻女子拿着一把粉红色的太阳伞"
}
```



### 3.2 sft — `utils/step3_generate_qa.py`

Qwen3.5 **看图**生成：每图 **2 问**（10 种描述问法随机 1 + 自拟细节问），答 ≤30/15 字 → `data/qa/coco_train_qa_qwen3.5.json`。支持断点续跑。

```bash
python utils/step3_generate_qa.py --num-images 10   # 试跑
python utils/step3_generate_qa.py --num-images 0    # 全量 train2014
```

生成完成后**停 vLLM**，再训练。

### 数据对比


|     | align              | sft                          |
| --- | ------------------ | ---------------------------- |
| 文件  | `coco_cn_qa.json`  | `coco_train_qa_qwen3.5.json` |
| 条数  | 20,341             | 165,566                      |
| 图片  | 20,341（COCO-CN 子集） | 82,783（train 全量）             |
| 问答  | 1 固定问 + 人工 caption | 2 问/图，Qwen3.5 生成             |


---



## Step 4：架构


| 模块        | 模型             | 训练     |
| --------- | -------------- | ------ |
| Vision    | SigLIP2-so400m | 冻结     |
| Projector | 2×MLP          | **训练** |
| LLM       | Qwen3-1.7B     | 冻结     |


`<image>` 占位符在序列中展开为 **576** 个视觉 token（384÷16=24 → 24×24 patch），与文本 embedding 拼接后送入 Qwen。训练时 `max_seq_len=704`（576 图 + 128 文本）。

**数据流举例**（demo 图 + sft 权重实测，B=1）：

```
question = "请用一句话描述这张图片。"
answer   = "客厅里，一位女子正在厨房里。"

prompt（训练时 answer 接在后面，truncate 至 128 token）：
  <|im_start|>system
  {system}
  
  <|im_start|>user
  <image>
  {question}
  
  <|im_start|>assistant
  {answer}

① 视觉路径
  pixel_values          (1, 3, 384, 384)
    → SigLIP Vision     (1, 576, 1152)
    → Projector         (1, 576, 2048)

② 文本 tokenize（T_text=46，含 1 个 <image> 占位 id）
  input_ids             (46,)
    → embed_tokens      (46, 2048)

③ 在 pos=22 处展开 <image>
  prefix  text[:22]     (22, 2048)     system + user 段至 <image> 前
  image   576 视觉 token (576, 2048)  替换 1 个 <image> id
  suffix  text[23:]      (23, 2048)    问题 + assistant 头 + answer
  ────────────────────────────────────
  merged  cat 沿 seq 维  (621, 2048)   621 = 46 - 1 + 576

④ 送入 Qwen3 LLM
  inputs_embeds         (1, 621, 2048)
  labels                (1, 621)       前 611 为 -100，后 10 个 answer 算 loss
    → Causal LM         (1, 621, 151670)

推理（仅 prompt）时 T_text=36 → T_seq=611，generate 再追加新 token。
```

验证：`python models/vlms/VLM_v1_model.py`（需 GPU；vLLM 占用显存时需先停服务）

---



## Step 5：训练

prompt labels = `-100`，只对 answer 算 loss；`max_seq_len=704`（576 图 + 128 文本）。

```bash
conda activate vlm && cd VLM

# align
python scripts/step5_train_vlm_v1.py --phase align

# sft（默认加载 align 的 projector.pt）
python scripts/step5_train_vlm_v1.py --phase sft
```


| 阶段    | 数据                                   | 输出                                      |
| ----- | ------------------------------------ | --------------------------------------- |
| align | `data/qa/coco_cn_qa.json`            | `checkpoints/VLM_v1_align/projector.pt` |
| sft   | `data/qa/coco_train_qa_qwen3.5.json` | `checkpoints/VLM_v1_sft/projector.pt`   |


日志：`logs/VLM_v1_<phase>/train.log`、`loss.png`。全量 sft 可后台：

```bash
mkdir -p logs/VLM_v1_sft
nohup python scripts/step5_train_vlm_v1.py --phase sft > logs/VLM_v1_sft/nohup.out 2>&1 &
```

---



## Step 6：命令行测试

```bash
python scripts/step6_test_vlm_v1.py                                                     # 默认 sft
python scripts/step6_test_vlm_v1.py --checkpoint checkpoints/VLM_v1_align/projector.pt  # 对比 align
```

---



## Step 7：Web 界面

```bash
conda activate vlm && cd VLM/web
python server.py
```

默认 `checkpoints/VLM_v1_sft/projector.pt`，端口 **7860**。上传图片后逐条提问，无多轮上下文。

**访问方式**


| 环境             | 做法                                                          |
| -------------- | ----------------------------------------------------------- |
| 本机 / 公司局域网     | 终端打印 `192.168.x.x:7860`，发给同网同事                              |
| AutoDL（Docker） | `172.17.x.x` 是容器 IP，**不能**当局域网用；在控制台做 **7860 端口映射**，用平台外链分享 |


---

