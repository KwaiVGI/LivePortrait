<h1 align="center">LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control</h1>

<div align='center'>
    <a href='https://github.com/cleardusk' target='_blank'><strong>郭建珠</strong></a><sup> 1†</sup>&emsp;
    <a href='https://github.com/Mystery099' target='_blank'><strong>张丁芸</strong></a><sup> 1,2</sup>&emsp;
    <a href='https://github.com/KwaiVGI' target='_blank'><strong>刘晓强</strong></a><sup> 1</sup>&emsp;
    <a href='https://github.com/zzzweakman' target='_blank'><strong>钟智舟</strong></a><sup> 1,3</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=_8k1ubAAAAAJ' target='_blank'><strong>张渊</strong></a><sup> 1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=P6MraaYAAAAJ' target='_blank'><strong>万鹏飞</strong></a><sup> 1</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Di_ZHANG3' target='_blank'><strong>张迪</strong></a><sup> 1</sup>&emsp;
</div>

<div align='center'>
    <sup>1 </sup>快手科技&emsp; <sup>2 </sup>中国科学技术大学&emsp; <sup>3 </sup>复旦大学&emsp;
</div>
<div align='center'>
    <small><sup>†</sup> 通讯作者（Project Lead）</small>
</div>

<br>
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/pdf/2407.03168'><img src='https://img.shields.io/badge/arXiv-LivePortrait-red'></a>
  <a href='https://liveportrait.github.io'><img src='https://img.shields.io/badge/Project-LivePortrait-green'></a>
  <a href='https://huggingface.co/spaces/KwaiVGI/liveportrait'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
  <a href="https://github.com/KwaiVGI/LivePortrait"><img src="https://img.shields.io/github/stars/KwaiVGI/LivePortrait"></a>
  <br>
  <a href="./readme.md"><strong>English</strong></a> | <strong>简体中文</strong>
</div>

<br>


<p align="center">
  <img src="./assets/docs/showcase2.gif" alt="showcase">
  <br>
  🔥 更多效果，请查看我们的 <a href="https://liveportrait.github.io/"><strong>主页</strong></a> 🔥
</p>



## 🔥 更新日志
- **`2024/10/18`**：⚠️ 我们更新了`transformers`，`gradio`库的版本避免安全漏洞，具体查看[这里](https://github.com/KwaiVGI/LivePortrait/pull/421).
- **`2024/08/29`**：📦 我们更新了Windows[一键安装程序](https://huggingface.co/cleardusk/LivePortrait-Windows/blob/main/LivePortrait-Windows-v20240829.zip)并支持自动更新, 详情建[这里](https://huggingface.co/cleardusk/LivePortrait-Windows#20240829)。
- **`2024/08/19`**：🖼️ 我们支持了**图像驱动模式**和**区域控制**。详情请见[**这里**](./assets/docs/changelog/2024-08-19.md)。
- **`2024/08/06`**：🎨 我们在Gradio界面支持**精确的人像编辑**, 受到[ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait)启发。详见[**这里**](./assets/docs/changelog/2024-08-06.md)。
- **`2024/08/05`**：📦 Windows用户现在可以下载[一键安装程序](https://huggingface.co/cleardusk/LivePortrait-Windows/blob/main/LivePortrait-Windows-v20240806.zip)，支持**人类模式**和**动物模式**！详情见[**这里**](./assets/docs/changelog/2024-08-05.md)。
- **`2024/08/02`**：😸 我们发布了**动物模型**版本，以及其他一些更新和改进。查看详情[**这里**](./assets/docs/changelog/2024-08-02.md)！
- **`2024/07/25`**：📦 Windows用户现在可以从 [HuggingFace](https://huggingface.co/cleardusk/LivePortrait-Windows/tree/main) 或 [百度云](https://pan.baidu.com/s/1FWsWqKe0eNfXrwjEhhCqlw?pwd=86q2) 下载软件包。解压并双击`run_windows.bat`即可享受！
- **`2024/07/24`**：🎨 我们在Gradio界面支持源人像的姿势编辑。我们还降低了默认检测阈值以增加召回率。[玩得开心](assets/docs/changelog/2024-07-24.md)！
- **`2024/07/19`**：✨ 我们支持🎞️ **人像视频编辑（aka v2v）**！更多信息见[**这里**](assets/docs/changelog/2024-07-19.md)。
- **`2024/07/17`**：🍎 我们支持macOS搭载Apple Silicon，修改来自 [jeethu](https://github.com/jeethu) 的PR [#143](https://github.com/KwaiVGI/LivePortrait/pull/143) 。
- **`2024/07/10`**：💪我们支持音频和视频拼接、驱动视频自动裁剪以及制作模板以保护隐私。更多信息见[这里](assets/docs/changelog/2024-07-10.md)。
- **`2024/07/09`**：🤗 我们发布了[HuggingFace Space](https://huggingface.co/spaces/KwaiVGI/liveportrait)，感谢HF团队和[Gradio](https://github.com/gradio-app/gradio)！
- **`2024/07/04`**：😊 我们发布了初始版本的推理代码和模型。持续更新，敬请关注！
- **`2024/07/04`**：🔥 我们发布了[主页](https://liveportrait.github.io)和在[arXiv](https://arxiv.org/pdf/2407.03168)上的技术报告。



## 介绍 📖
此仓库名为**LivePortrait**，包含我们论文（[LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://arxiv.org/pdf/2407.03168)）的官方PyTorch实现。 我们正在积极更新和改进此仓库。如果您发现任何错误或有建议，欢迎提出问题或提交合并请求💖。

## 上手指南 🏁
### 1. 克隆代码和安装运行环境 🛠️

> [!Note]
> 确保您的系统已安装[`git`](https://git-scm.com/)、[`conda`](https://anaconda.org/anaconda/conda)和[`FFmpeg`](https://ffmpeg.org/download.html)。有关FFmpeg安装的详细信息，见[**如何安装FFmpeg**](assets/docs/how-to-install-ffmpeg.md)。

```bash
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait

# 使用conda创建环境
conda create -n LivePortrait python=3.9
conda activate LivePortrait
```

#### 对于Linux或Windows用户

[X-Pose](https://github.com/IDEA-Research/X-Pose)需要您的`torch`版本与CUDA版本兼容。

首先，通过以下命令检查您当前的CUDA版本：

```bash
nvcc -V # example versions: 11.1, 11.8, 12.1, etc.
```

然后，安装相应版本的torch。以下是不同CUDA版本的示例。如果您的CUDA版本未列出，请访问[PyTorch官方网站](https://pytorch.org/get-started/previous-versions)获取安装命令：
```bash
# for CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# for CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# for CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# ...
```

**注意**：在Windows系统上，一些过高版本的CUDA(12.4、12.6等)可能会导致未知的问题，您可以考虑降低您的CUDA版本到11.8，这是我们测试的一个较为稳定的版本。降级方法可以参考 [@dimitribarbot](https://github.com/dimitribarbot)
提供的[文档](https://github.com/dimitribarbot/sd-webui-live-portrait/blob/main/assets/docs/how-to-install-xpose.md#cuda-toolkit-118).

最后，安装其余依赖项：

```bash
pip install -r requirements.txt
```

#### 对于搭载Apple Silicon的macOS用户

[X-Pose](https://github.com/IDEA-Research/X-Pose)依赖项不支持macOS，因此您可以跳过其安装。人类模式照常工作，但不支持动物模式。使用为搭载Apple Silicon的macOS提供的requirements文件：

```bash
# 对于搭载Apple Silicon的macOS用户
pip install -r requirements_macOS.txt
```

### 2. 下载预训练权重(Pretrained weights) 📥

从HuggingFace下载预训练权重的最简单方法是：
```bash
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

若您不能访问HuggingFace平台，你可以访问其镜像网站[hf-mirror](https://hf-mirror.com/)进行下载操作：

```bash
# !pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

或者，您可以从[Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib)或[百度云](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn)（进行中）下载所有预训练权重。解压并将它们放置在`./pretrained_weights`目录下。

确保目录结构如所示包含[**本仓库该路径**](assets/docs/directory-structure.md)其中展示的内容。

### 3. 推理 🚀

#### 快速上手（人类模型）👤

```bash
# 对于Linux和Windows用户
python inference.py

# 对于搭载Apple Silicon的macOS用户（Intel未测试）。注意：这可能比RTX 4090慢20倍
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py
```

如果脚本成功运行，您将得到一个名为`animations/s6--d0_concat.mp4`的输出mp4文件。此文件包含以下结果：驱动视频、输入图像或视频以及生成结果。

<p align="center">
  <img src="./assets/docs/inference.gif" alt="image">
</p>
或者，您可以通过指定`-s`和`-d`参数来更改输入：

```bash
# 源输入是图像
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4

# 源输入是视频 ✨
python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d0.mp4

# 更多选项请见
python inference.py -h
```

#### 快速上手（动物模型） 🐱🐶

动物模式仅在Linux和Windows上经过测试，并且需要NVIDIA GPU。

您需要首先构建一个名为`MultiScaleDeformableAttention`的OP，该OP由[X-Pose](https://github.com/IDEA-Research/X-Pose)使用，这是一个通用的关键点检测框架。

```bash
cd src/utils/dependencies/XPose/models/UniPose/ops
python setup.py build install
cd - # 等同于 cd ../../../../../../../
```

然后执行
```bash
python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/wink.pkl --driving_multiplier 1.75 --no_flag_stitching
```
如果脚本成功运行，您将得到一个名为`animations/s39--wink_concat.mp4`的输出mp4文件。
<p align="center">
  <img src="./assets/docs/inference-animals.gif" alt="image">
</p>

#### 驱动视频自动裁剪 📢📢📢

> [!IMPORTANT]
> 使用您自己的驱动视频时，我们**推荐**： ⬇️
>
> - 将其裁剪为**1:1**的宽高比（例如，512x512或256x256像素），或通过`--flag_crop_driving_video`启用自动裁剪。
> - 专注于头部区域，类似于示例视频。
> - 最小化肩部运动。
> - 确保驱动视频的第一帧是具有**中性表情**的正面面部。

以下是通过`--flag_crop_driving_video`自动裁剪的示例：

```bash
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d13.mp4 --flag_crop_driving_video
```

如果自动裁剪的结果不理想，您可以修改`--scale_crop_driving_video`、`--vy_ratio_crop_driving_video`选项来调整比例和偏移，或者手动进行调整。

#### 动作模板制作

您也可以使用以`.pkl`结尾的自动生成的动作模板文件来加快推理速度，并**保护隐私**，例如：
```bash
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d5.pkl # 人像动画
python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d5.pkl # 人像视频编辑
```

### 4. Gradio 界面 🤗

我们还提供了Gradio界面 <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>，以获得更好的体验，只需运行：

```bash
# 对于Linux和Windows用户（以及搭载Intel的macOS？？）
python app.py # 人类模型模式

# 对于搭载Apple Silicon的macOS用户，不支持Intel，这可能比RTX 4090慢20倍
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py # 人类模型模式
```

我们还为动物模式提供了Gradio界面，这仅在Linux上经过NVIDIA GPU测试：
```bash
python app_animals.py # animals mode 🐱🐶
```

您可以指定`--server_port`、`--share`、`--server_name`参数以满足您的需求！

🚀我们还提供了一个加速选项`--flag_do_torch_compile`。第一次推理触发优化过程（约一分钟），使后续推理速度提高20-30%。不同CUDA版本的性能提升可能有所不同。

```bash
# 启用torch.compile以进行更快的推理
python app.py --flag_do_torch_compile
```
**注意**：此方法在Windows和macOS上不受支持。

**或者，在[HuggingFace](https://huggingface.co/spaces/KwaiVGI/LivePortrait)上轻松尝试**🤗。

### 5. 推理速度预估 🚀🚀🚀
我们还提供了一个脚本来评估每个模块的推理速度：

```bash
# 对于NVIDIA GPU
python speed.py
```

结果在[**本仓库该文件展示**](./assets/docs/speed.md).

## 社区资源 🤗

### 社区项目

| 仓库 (按时间排序) | 描述 | 作者 |
|------|------|--------|
| [**FacePoke**](https://github.com/jbilcke-hf/FacePoke) | 一个实时的头部姿态表情控制应用，通过鼠标控制！ | [@jbilcke-hf](https://github.com/jbilcke-hf) |
| [**FaceFusion**](https://github.com/facefusion/facefusion) | FaceFusion 3.0 集成了 LivePortrait 作为 `expression_restorer` 和 `face_editor` 处理器。 | [@henryruhs](https://github.com/henryruhs) |
| [**sd-webui-live-portrait**](https://github.com/dimitribarbot/sd-webui-live-portrait) | LivePortrait 的 WebUI 扩展，在原版 Stable Diffusion WebUI 中添加了一个标签以使用 LivePortrait 的功能。 | [@dimitribarbot](https://github.com/dimitribarbot) |
| [**ComfyUI-LivePortraitKJ**](https://github.com/kijai/ComfyUI-LivePortraitKJ) | 一个用于 LivePortrait 的 ComfyUI 节点，使用 MediaPipe 作为 Insightface 的替代方案。 | [@kijai](https://github.com/kijai) |
| [**ComfyUI-AdvancedLivePortrait**](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait) | 一个更快的 ComfyUI 节点，具有实时预览功能，启发了许多社区开发的工具和项目。 | [@PowerHouseMan](https://github.com/PowerHouseMan) |
| [**comfyui-liveportrait**](https://github.com/shadowcz007/comfyui-liveportrait) | 一个用于 LivePortrait 的 ComfyUI 节点，支持多面部、表情插值等功能，并有[教程](https://www.bilibili.com/video/BV1JW421R7sP)。 | [@shadowcz007](https://github.com/shadowcz007) |

### Playgrounds, 🤗 HuggingFace Spaces 以及其它
- [FacePoke Space](https://huggingface.co/spaces/jbilcke-hf/FacePoke)
- [Expression Editor Space](https://huggingface.co/spaces/fffiloni/expression-editor)
- [Expression Editor Replicate](https://replicate.com/fofr/expression-editor)
- [Face Control Realtime Demo](https://fal.ai/demos/face-control) on FAL
- [Replicate Playground](https://replicate.com/fofr/live-portrait)
- Nuke 可以通过 CompyUI 节点使用 LivePortrait，详情见[这里](https://x.com/bilawalsidhu/status/1837349806475276338)
- LivePortrait 在 [Poe](https://poe.com/LivePortrait) 上运行

### 视频教程
- [LivePortrait 视频转视频的工作流程](https://youtu.be/xfzK_6cTs58?si=aYjgypeJBkhc46VL) 由 [@curiousrefuge](https://www.youtube.com/@curiousrefuge) 制作
- [Google Colab 教程](https://youtu.be/59Y9ePAXTp0?si=KzEWhklBlporW7D8) 由 [@Planet Ai](https://www.youtube.com/@planetai217) 制作
- [论文解读](https://youtu.be/fD0P6UWSu8I?si=Vn5wxUa8qSu1jv4l) 由 [@TwoMinutePapers](https://www.youtube.com/@TwoMinutePapers) 制作
- [ComfyUI 高级 LivePortrait 教程](https://youtu.be/q0Vf-ZZsbzI?si=nbs3npleH-dVCt28) 由 [TutoView](https://www.youtube.com/@TutoView) 制作
- [LivePortrait 探索](https://www.youtube.com/watch?v=vsvlbTEqgXQ) 和 [LivePortrait 深入探讨](https://youtu.be/cucaEEDYmsw?si=AtPaDWc5G-a4E8dD) 由 [TheoreticallyMedia](https://www.youtube.com/@TheoreticallyMedia) 制作
- [LivePortrait 实战教程](https://www.youtube.com/watch?v=uyjSTAOY7yI) 由 [@AI Search](https://www.youtube.com/@theAIsearch) 制作
- [ComfyUI 教程](https://www.youtube.com/watch?v=8-IcDDmiUMM) 由 [@Sebastian Kamph](https://www.youtube.com/@sebastiankamph) 制作
- B 站上的[教程](https://www.bilibili.com/video/BV1cf421i7Ly)

还有来自社区的无数令人惊叹的贡献，未能一一列举 💖

## 致谢 💐

我们要感谢[FOMM](https://github.com/AliaksandrSiarohin/first-order-model)、[Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)、[SPADE](https://github.com/NVlabs/SPADE)、[InsightFace](https://github.com/deepinsight/insightface)和[X-Pose](https://github.com/IDEA-Research/X-Pose)仓库的的贡献者，感谢他们的开放研究和贡献。

## 道德考量 🛡️
肖像动画技术伴随着社会风险，特别是在创建深度伪造（deepfakes）时可能被滥用。为了减轻这些风险，遵循道德指南并采取负责任的使用实践至关重要。目前，生成的结果包含一些视觉伪影，这些伪影可能有助于检测深度伪造。请注意，我们不对本项目生成的结果的使用承担任何法律责任。

## 引用 💖

如果您发现LivePortrait对您的研究有用，欢迎引用我们的工作，使用以下BibTeX：

```bibtex
@article{guo2024liveportrait,
  title   = {LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control},
  author  = {Guo, Jianzhu and Zhang, Dingyun and Liu, Xiaoqiang and Zhong, Zhizhou and Zhang, Yuan and Wan, Pengfei and Zhang, Di},
  journal = {arXiv preprint arXiv:2407.03168},
  year    = {2024}
}
```

## 联系方式 📧

[**Jianzhu Guo (郭建珠)**](https://guojianzhu.com); **guojianzhu1994@gmail.com**；
