# 常用命令

## 磁盘占用

```bash
# VLM 项目目录
du -h --max-depth=2 /root/autodl-fs/VLM | sort -hr

# 系统盘
df -h /

# /root 各子目录
du -h --max-depth=1 /root | sort -hr

# 统计 COCO 图片数量
find data/COCO2017/val2017 -type f -name "*.jpg" | wc -l
```

## conda 环境

```bash
# 删除环境
conda env remove -n vlm

# 创建环境
conda create -n vlm python=3.12 -y
conda activate vlm

# 查看环境
conda env list
```

## pip 安装

```bash
cd /root/autodl-fs/VLM
pip config set global.cache-dir $(pwd)/.pip-cache
pip install -r requirements.txt

# numpy 下载超时，单独安装
mkdir -p wheels
pip download numpy==2.3.5 -d wheels \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --python-version 3.12 --only-binary=:all:
pip install wheels/numpy-*.whl
pip install -r requirements.txt
```

## 验证环境

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 监控安装进度

```bash
# pip 是否在运行
ps aux | grep "pip install" | grep -v grep

# env 目录大小
watch -n 5 'du -sh /root/miniconda3/envs/vlm'

# pip 当前解压目录
ls -lt /tmp/pip-unpack-* | head -3
du -sh /tmp/pip-unpack-*
```

## 系统盘清理

```bash
# pip 缓存（装完后再清）
pip cache purge

# 旧 ModelScope 缓存
rm -rf ~/.cache/modelscope

# conda 包缓存
conda clean -a -y

# 临时文件
rm -rf /tmp/vllm_check
```

## ModelScope 模型下载

```bash
mkdir -p models
modelscope download --model Qwen/Qwen3.5-9B --local_dir models/Qwen3.5-9B
modelscope download --model google/siglip2-so400m-patch16-384 --local_dir models/siglip2-so400m-patch16-384
modelscope download --model Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B
```
