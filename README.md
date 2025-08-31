# 图像打标工具使用指南

## 目录
- [前置准备](#前置准备)
- [环境搭建](#环境搭建)
- [依赖安装](#依赖安装)
- [API配置](#api配置)
- [使用方法](#使用方法)

## 前置准备

### Git代理设置（可选）
如果git克隆操作无法正常使用，可能需要配置代理：
1. 找到VPN的代理端口
2. 执行以下命令设置git代理：
   ```bash
   git config --global http.proxy http://127.0.0.1:代理端口
   git config --global https.proxy https://127.0.0.1:代理端口
   ```

## 环境搭建

### 创建并激活虚拟环境
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows系统
venv\Scripts\activate
# macOS/Linux系统
source venv/bin/activate

# 更新pip
python.exe -m pip install --upgrade pip
```

### 下载模型
```bash
python florence-model.py
```

## 依赖安装

### 根据CUDA版本安装PyTorch及相关库

根据您系统中的CUDA版本，选择以下对应命令执行：

```bash
# 首先检查CUDA版本
nvcc --version

# 根据CUDA版本选择安装命令
# CUDA主版本≥12的安装逻辑
if (( cuda_major_version >= 12 )); then
    echo "install torch 2.7.0+cu128"
    pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    pip install --no-deps xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8及以上的安装逻辑
elif (( cuda_major_version == 11 && cuda_minor_version >= 8 )); then
    echo "install torch 2.4.0+cu118"
    pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    pip install --no-deps xformers==0.0.27.post2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 11.6及以上的安装逻辑
elif (( cuda_major_version == 11 && cuda_minor_version >= 6 )); then
    echo "install torch 1.12.1+cu116"
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install --upgrade git+https://github.com/facebookresearch/xformers.git@0bad001ddd56c080524d37c84ff58d9cd030ebfd
    pip install triton==2.0.0.dev20221202

# CUDA 11.2及以上的安装逻辑
elif (( cuda_major_version == 11 && cuda_minor_version >= 2 )); then
    echo "install torch 1.12.1+cu113"
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install --upgrade git+https://github.com/facebookresearch/xformers.git@0bad001ddd56c080524d37c84ff58d9cd030ebfd
    pip install triton==2.0.0.dev20221202
fi
```

### 安装项目依赖
```bash
pip install -r requirements.txt
pip install -r requirements-db.txt
```

## API配置

设置标签编辑器翻译API：
1. 打开`app.py`文件
2. 找到百度翻译API配置部分，替换为您自己的appid和appkey：
   ```python
   # 百度翻译API配置 - 请替换为您自己的appid和appkey
   BAIDU_APPID = '你的appid'
   BAIDU_APPKEY = '你的appkey'
   ```

## 使用方法

### 测试为图片打标（Flux模型）
```bash
python florence-tag.py
```

### 为SD模型打标
```bash
# 安装必要依赖
pip install onnx onnxruntime-gpu

# 如提示缺少cmake，请先从https://cmake.org/download/下载安装

# 运行打标脚本
python sd-tag.py
```
