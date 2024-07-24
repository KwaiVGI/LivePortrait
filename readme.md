<h1 align="center">LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control</h1>

<div align='center'>
    <a href='https://github.com/cleardusk' target='_blank'><strong>Jianzhu Guo</strong></a><sup> 1†</sup>&emsp;
    <a href='https://github.com/KwaiVGI' target='_blank'><strong>Dingyun Zhang</strong></a><sup> 1,2</sup>&emsp;
    <a href='https://github.com/KwaiVGI' target='_blank'><strong>Xiaoqiang Liu</strong></a><sup> 1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=t88nyvsAAAAJ&hl' target='_blank'><strong>Zhizhou Zhong</strong></a><sup> 1,3</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=_8k1ubAAAAAJ' target='_blank'><strong>Yuan Zhang</strong></a><sup> 1</sup>&emsp;
</div>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=P6MraaYAAAAJ' target='_blank'><strong>Pengfei Wan</strong></a><sup> 1</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Di_ZHANG3' target='_blank'><strong>Di Zhang</strong></a><sup> 1</sup>&emsp;
</div>

<div align='center'>
    <sup>1 </sup>Kuaishou Technology&emsp; <sup>2 </sup>University of Science and Technology of China&emsp; <sup>3 </sup>Fudan University&emsp;
</div>
<div align='center'>
    <small><sup>†</sup> Corresponding author</small>
</div>

<br>
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/pdf/2407.03168'><img src='https://img.shields.io/badge/arXiv-LivePortrait-red'></a>
  <a href='https://liveportrait.github.io'><img src='https://img.shields.io/badge/Project-LivePortrait-green'></a>
  <a href='https://huggingface.co/spaces/KwaiVGI/liveportrait'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
  <a href="https://github.com/KwaiVGI/LivePortrait"><img src="https://img.shields.io/github/stars/KwaiVGI/LivePortrait"></a>
</div>
<br>

<p align="center">
  <img src="./assets/docs/showcase2.gif" alt="showcase">
  <br>
  🔥 For more results, visit our <a href="https://liveportrait.github.io/"><strong>homepage</strong></a> 🔥
</p>



## 🔥 Updates
- **`2024/07/24`**: 🎨 We support pose editing for source portraits in the Gradio interface. We've also lowered the default detection threshold to support more input detections. [Have fun](assets/docs/changelog/2024-07-24.md)!
- **`2024/07/19`**: ✨ We support 🎞️ **portrait video editing (aka v2v)**! More to see [here](assets/docs/changelog/2024-07-19.md).
- **`2024/07/17`**: 🍎 We support macOS with Apple Silicon, modified from [jeethu](https://github.com/jeethu)'s PR [#143](https://github.com/KwaiVGI/LivePortrait/pull/143).
- **`2024/07/10`**: 💪 We support audio and video concatenating, driving video auto-cropping, and template making to protect privacy. More to see [here](assets/docs/changelog/2024-07-10.md).
- **`2024/07/09`**: 🤗 We released the [HuggingFace Space](https://huggingface.co/spaces/KwaiVGI/liveportrait), thanks to the HF team and [Gradio](https://github.com/gradio-app/gradio)!
- **`2024/07/04`**: 😊 We released the initial version of the inference code and models. Continuous updates, stay tuned!
- **`2024/07/04`**: 🔥 We released the [homepage](https://liveportrait.github.io) and technical report on [arXiv](https://arxiv.org/pdf/2407.03168).



## Introduction 📖
This repo, named **LivePortrait**, contains the official PyTorch implementation of our paper [LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://arxiv.org/pdf/2407.03168).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

## Getting Started 🏁
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait

# create env using conda
conda create -n LivePortrait python=3.9
conda activate LivePortrait

# install dependencies with pip
# for Linux and Windows users
pip install -r requirements.txt
# for macOS with Apple Silicon users
pip install -r requirements_macOS.txt
```

**Note:** make sure your system has [FFmpeg](https://ffmpeg.org/download.html) installed, including both `ffmpeg` and `ffprobe`!

### 2. Download pretrained weights

The easiest way to download the pretrained weights is from HuggingFace:
```bash
# first, ensure git-lfs is installed, see: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
git lfs install
# clone and move the weights
git clone https://huggingface.co/KwaiVGI/LivePortrait temp_pretrained_weights
mv temp_pretrained_weights/* pretrained_weights/
rm -rf temp_pretrained_weights
```

Alternatively, you can download all pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib) or [Baidu Yun](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn). Unzip and place them in `./pretrained_weights`.

Ensuring the directory structure is as follows, or contains:
```text
pretrained_weights
├── insightface
│   └── models
│       └── buffalo_l
│           ├── 2d106det.onnx
│           └── det_10g.onnx
└── liveportrait
    ├── base_models
    │   ├── appearance_feature_extractor.pth
    │   ├── motion_extractor.pth
    │   ├── spade_generator.pth
    │   └── warping_module.pth
    ├── landmark.onnx
    └── retargeting_models
        └── stitching_retargeting_module.pth
```

### 3. Inference 🚀

#### Fast hands-on
```bash
# For Linux and Windows
python inference.py

# For macOS with Apple Silicon, Intel not supported, this maybe 20x slower than RTX 4090
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py
```

If the script runs successfully, you will get an output mp4 file named `animations/s6--d0_concat.mp4`. This file includes the following results: driving video, input image or video, and generated result.

<p align="center">
  <img src="./assets/docs/inference.gif" alt="image">
</p>

Or, you can change the input by specifying the `-s` and `-d` arguments:

```bash
# source input is an image
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4

# source input is a video ✨
python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d0.mp4

# more options to see
python inference.py -h
```

#### Driving video auto-cropping 📢📢📢
To use your own driving video, we **recommend**: ⬇️
 - Crop it to a **1:1** aspect ratio (e.g., 512x512 or 256x256 pixels), or enable auto-cropping by `--flag_crop_driving_video`.
 - Focus on the head area, similar to the example videos.
 - Minimize shoulder movement.
 - Make sure the first frame of driving video is a frontal face with **neutral expression**.

Below is a auto-cropping case by `--flag_crop_driving_video`:
```bash
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d13.mp4 --flag_crop_driving_video
```

If you find the results of auto-cropping is not well, you can modify the `--scale_crop_driving_video`, `--vy_ratio_crop_driving_video` options to adjust the scale and offset, or do it manually.

#### Motion template making
You can also use the auto-generated motion template files ending with `.pkl` to speed up inference, and **protect privacy**, such as:
```bash
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d5.pkl # portrait animation
python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d5.pkl # portrait video editing
```

### 4. Gradio interface 🤗

We also provide a Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> interface for a better experience, just run by:

```bash
# For Linux and Windows users (and macOS with Intel??)
python app.py

# For macOS with Apple Silicon users, Intel not supported, this maybe 20x slower than RTX 4090
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py
```

You can specify the `--server_port`, `--share`, `--server_name` arguments to satisfy your needs!

🚀 We also provide an acceleration option `--flag_do_torch_compile`. The first-time inference triggers an optimization process (about one minute), making subsequent inferences 20-30% faster. Performance gains may vary with different CUDA versions.
```bash
# enable torch.compile for faster inference
python app.py --flag_do_torch_compile
```
**Note**: This method is not supported on Windows and macOS.

**Or, try it out effortlessly on [HuggingFace](https://huggingface.co/spaces/KwaiVGI/LivePortrait) 🤗**

### 5. Inference speed evaluation 🚀🚀🚀
We have also provided a script to evaluate the inference speed of each module:

```bash
# For NVIDIA GPU
python speed.py
```

Below are the results of inferring one frame on an RTX 4090 GPU using the native PyTorch framework with `torch.compile`:

| Model                             | Parameters(M) | Model Size(MB) | Inference(ms) |
|-----------------------------------|:-------------:|:--------------:|:-------------:|
| Appearance Feature Extractor      |     0.84      |       3.3      |     0.82      |
| Motion Extractor                  |     28.12     |       108      |     0.84      |
| Spade Generator                   |     55.37     |       212      |     7.59      |
| Warping Module                    |     45.53     |       174      |     5.21      |
| Stitching and Retargeting Modules |     0.23      |       2.3      |     0.31      |

*Note: The values for the Stitching and Retargeting Modules represent the combined parameter counts and total inference time of three sequential MLP networks.*

## Community Resources 🤗

Discover the invaluable resources contributed by our community to enhance your LivePortrait experience:

- [ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ) by [@kijai](https://github.com/kijai)
- [comfyui-liveportrait](https://github.com/shadowcz007/comfyui-liveportrait) by [@shadowcz007](https://github.com/shadowcz007)
- [LivePortrait In ComfyUI](https://www.youtube.com/watch?v=aFcS31OWMjE) by [@Benji](https://www.youtube.com/@TheFutureThinker)
- [LivePortrait hands-on tutorial](https://www.youtube.com/watch?v=uyjSTAOY7yI) by [@AI Search](https://www.youtube.com/@theAIsearch)
- [ComfyUI tutorial](https://www.youtube.com/watch?v=8-IcDDmiUMM) by [@Sebastian Kamph](https://www.youtube.com/@sebastiankamph)
- [Replicate Playground](https://replicate.com/fofr/live-portrait) and [cog-comfyui](https://github.com/fofr/cog-comfyui) by [@fofr](https://github.com/fofr)

And many more amazing contributions from our community!

## Acknowledgements 💐
We would like to thank the contributors of [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [SPADE](https://github.com/NVlabs/SPADE), [InsightFace](https://github.com/deepinsight/insightface) repositories, for their open research and contributions.

## Citation 💖
If you find LivePortrait useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{guo2024liveportrait,
  title   = {LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control},
  author  = {Guo, Jianzhu and Zhang, Dingyun and Liu, Xiaoqiang and Zhong, Zhizhou and Zhang, Yuan and Wan, Pengfei and Zhang, Di},
  journal = {arXiv preprint arXiv:2407.03168},
  year    = {2024}
}
```
