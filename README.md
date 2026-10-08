# WorldCrafter: Consistent Video World Model with Implicit 3D-aware Memory

<p align="center">
  <a href="https://drexubery.github.io/">Wangbo Yu</a><sup>1*</sup>,
  <a href="https://kunhao-liu.github.io/">Kunhao Liu</a><sup>1*</sup>,
  <a href="https://wbhu.github.io/">Wenbo Hu</a><sup>1†</sup>,
  <a href="https://shyuanbest.github.io/">Shenghai Yuan</a><sup>2</sup>,
  <a href="https://www.falcary.com/">Chaoran Feng</a><sup>2</sup>,
  <a href="https://github.com/zhouhyOcean">Haiyang Zhou</a><sup>2</sup><br>
  <a href="https://yukun-huang.github.io/">Yukun Huang</a><sup>1</sup>,
  <a href="https://raymondwang987.github.io/">Yiran Wang</a><sup>1</sup>,
  <a href="https://thuzhaowang.github.io/">Wang Zhao</a><sup>1</sup>,
  <a href="https://huggingface.co/luoyingmin">Yingmin Luo</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&amp;hl=en">Ying Shan</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>ARC Lab, Tencent IEG &nbsp; <sup>2</sup>Peking University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.24984"><img src="https://img.shields.io/badge/arXiv-2609.24984-b31b1b.svg" alt="arXiv Paper"></a> &nbsp;
  <a href="https://drexubery.github.io/WorldCrafter/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;
  <a href="https://www.youtube.com/watch?v=sg09ftQOl0E&amp;t=5s"><img src="https://img.shields.io/badge/Youtube-Video-b31b1b.svg" alt="YouTube Video"></a> &nbsp;
  <a href="https://huggingface.co/spaces/Drexubery/worldcrafter-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Hugging Face Demo"></a>
</p>

🤗 If you find WorldCrafter useful, please consider giving this repo a ⭐. Your support helps us share and improve the project. Thank you!

## 🔆 Introduction

WorldCrafter enables consistent, camera-controlled scene exploration from an image or text prompt. Its camera-queryable implicit 3D-aware memory preserves scene information across viewpoints and over long horizons.

We provide **[WorldCrafter-Base](https://huggingface.co/TencentARC/WorldCrafter-Base)** and **[WorldCrafter-Fast](https://huggingface.co/TencentARC/WorldCrafter-Fast)**, a distilled model for faster inference.

🎮 **Our interactive demo code and serving infrastructure are fully open source**, enabling the community to run, customize, and build on WorldCrafter. See **[Interactive Demo](#-interactive-demo)** to get started.

https://github.com/user-attachments/assets/721a6e31-e411-4802-8c60-3ccc6e9cb33b

## ⚙️ Setup

### 1. Clone WorldCrafter

```bash
git clone https://github.com/TencentARC/WorldCrafter.git
cd WorldCrafter
```

### 2. Environment

Set up the environment with **uv** or **conda + pip**. Both methods use Python 3.11 on Linux and require an NVIDIA GPU with a compatible driver.

**A: uv (recommended)**

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

uv sync --project uvenv --frozen --extra demo
source uvenv/.venv/bin/activate
```

For other Linux distributions, install [FFmpeg](https://ffmpeg.org/download.html)
using your system package manager.

This installs the locked PyTorch 2.10 / CUDA 12.8 environment and its
acceleration dependencies.

**B: conda + pip**

Create an environment and install PyTorch for your machine. For CUDA 12.8:

```bash
conda create -n worldcrafter -c conda-forge python=3.11 pip ffmpeg -y
conda activate worldcrafter
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[demo,xformers]" flash-attn-3==3.0.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Choose the appropriate CUDA build from the
[PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2100).

### 3. Model weights

| Models | Download Link | Notes |
| --- | --- | --- |
| WorldCrafter-Base | 🤗 [Hugging Face](https://huggingface.co/TencentARC/WorldCrafter-Base) | Base model |
| WorldCrafter-Fast | 🤗 [Hugging Face](https://huggingface.co/TencentARC/WorldCrafter-Fast) | Distilled high- and low-noise models for faster inference |

Download weights with the Hugging Face CLI:

```bash
hf download TencentARC/WorldCrafter-Fast --local-dir weights/WorldCrafter-Fast

# Optional: also download Base to run the base model
hf download TencentARC/WorldCrafter-Base --local-dir weights/WorldCrafter-Base

# Optional: generate prompts automatically from input images
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir weights/Qwen3-VL-4B-Instruct

```

Base model uses shared components from `WorldCrafter-Fast`, so keep both folders when using base model.

## 💫 Inference

See the [inference guide](test/README.md) for camera controls, prompt writing, examples and custom inputs.

### 1. Image-to-video

Run with the Base or distilled Fast model:

```bash
# Base
python inference.py --model-type base --mode i2v \
  --image-path test/I2V/00_cat_robot_vacuum/image.png \
  --prompt test/I2V/00_cat_robot_vacuum/prompt.txt \
  --camera-path test/I2V/00_cat_robot_vacuum/camera.npy

# Fast
python inference.py --model-type fast --mode i2v \
  --image-path test/I2V/03_waterfall/image.png \
  --prompt test/I2V/03_waterfall/prompt.txt \
  --camera-path test/I2V/03_waterfall/camera.npy
```


`--prompt` accepts text or a `.txt` file. For a custom input image, use `--prompt auto-first-person`
for a first-person view scene description or `--prompt auto-third-person` for a third-person view scene description.
This uses [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
to automatically write the prompt. For example:

```bash
# Base
python inference.py --model-type base --mode i2v \
  --image-path test/I2V/00_cat_robot_vacuum/image.png \
  --prompt auto-third-person \
  --camera-path test/I2V/00_cat_robot_vacuum/camera.npy

# Fast
python inference.py --model-type fast --mode i2v \
  --image-path test/I2V/03_waterfall/image.png \
  --prompt auto-first-person \
  --camera-path test/I2V/03_waterfall/camera.npy
```

### 2. Text-to-video

```bash
# Base
python inference.py --model-type base --mode t2v \
  --prompt test/T2V/00_red_balloon/prompt.txt \
  --camera-path test/T2V/00_red_balloon/camera.npy

# Fast
python inference.py --model-type fast --mode t2v \
  --prompt test/T2V/02_tokyo_street/prompt.txt \
  --camera-path test/T2V/02_tokyo_street/camera.npy
```

Compilation is **off by default**. Add `--enable-compile` to enable it; the first run takes longer to start.



## 🎮 Interactive Demo

Explore a scene from an image with keyboard camera controls, powered by
WorldCrafter-Fast. Run from the repository root with your environment activated:

```bash
python -m demo --model-path weights/WorldCrafter-Fast
```

Open `http://localhost:8080` in your browser. Compilation is enabled by
default, so the first generation takes longer. Add `--devices 0,1` to
run on two GPUs.

See the [demo guide](demo/README.md) for camera controls, automatic
prompt generation, and deployment options.

## 📝 Citation

If you find WorldCrafter useful in your research, please cite:

```bibtex
@article{yu2026worldcrafter,
  title={WorldCrafter: Consistent Video World Model with Implicit {3D}-aware Memory},
  author={Yu, Wangbo and Liu, Kunhao and Hu, Wenbo and Yuan, Shenghai and Feng, Chaoran and Zhou, Haiyang and Huang, Yukun and Wang, Yiran and Zhao, Wang and Luo, Yingmin and Shan, Ying},
  journal={arXiv preprint arXiv:2609.24984},
  year={2026}
}
```

## 📄 License

See [LICENSE.txt](LICENSE.txt) for the terms of use and third-party attributions.

## 🤗 Related Works

[Helios](https://github.com/PKU-YuanGroup/Helios),
[LagerNVS](https://github.com/facebookresearch/lagernvs),
[DreamX-World](https://github.com/AMAP-ML/DreamX-World),
[EVOKE](https://github.com/AlayaLab/Evoke),
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay),
[Lyra 2.0](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2),
[Echo-WM](https://github.com/jd-opensource/JoyAI-Echo/tree/main/echo_wm),
[LingBot-World 2](https://github.com/robbyant/lingbot-world-v2),
[Matrix-Game 3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5),
[SANA-WM](https://github.com/NVlabs/Sana/blob/main/docs/sana_wm.md).
