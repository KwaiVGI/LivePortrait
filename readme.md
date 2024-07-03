<h1 align="center">LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control</h1>

<div align='center'>
    <a href='https://github.com/cleardusk' target='_blank'>Jianzhu Guo</a><sup>1*</sup>&emsp;
    <a href='https://github.com/KwaiVGI' target='_blank'>Dingyun Zhang</a><sup>1,2</sup>&emsp;
    <a href='https://github.com/KwaiVGI' target='_blank'>Xiaoqiang Liu</a><sup>1</sup>&emsp;
    <a href='https://github.com/KwaiVGI' target='_blank'>Zhizhou Zhong</a><sup>1,3</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=_8k1ubAAAAAJ' target='_blank'>Yuan Zhang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=P6MraaYAAAAJ' target='_blank'>Pengfei Wan</a><sup>1</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Di_ZHANG3' target='_blank'>Di Zhang</a><sup>1</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Kuaishou Technology&emsp; <sup>2</sup>University of Science and Technology of China&emsp; <sup>3</sup>Fudan University&emsp;
</div>

<br>
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://liveportrait.github.io'><img src='https://img.shields.io/badge/Project-Homepage-green'></a>
  <a href='https://github.com/KwaiVGI/LivePortrait'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
</div>
<br>

<p align="center">
  <img src="./assets/docs/showcase2.gif" alt="showcase">
</p>



## 🔥 Updates
- **`2024/07/04`**: 🔥 We released the initial version of the inference code and models.
- **`2024/07/04`**: 😊 We released the technique report on [arXiv]().

## Introduction
This repo, named **LivePortrait**, contains the official PyTorch implementation of our paper [LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control]().
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

## 🔥 Getting Started
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait
# using lfs to pull the data
git lfs install
git lfs pull

# create env using conda
conda create -n LivePortrait python==3.9.18
conda activate LivePortrait
# install dependencies with pip
pip install -r requirements.txt
```

### 2. Download pretrained weights
Download our pretrained LivePortrait weights and face detection models of InsightFace from [Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib) or [Baidu Yun](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn). We have packed all weights in one directory 😊. Unzip and place them in `./pretrained_weights` ensuring the directory structure is as follows:
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

```bash
python inference.py
```

If the script runs successfully, you will see the following results: driving video, input image, and generated result.

<p align="center">
  <img src="./assets/docs/inference.gif" alt="image">
</p>

Or, you can change the input by specifying the `-s` and `-d` arguments:

```bash
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4

# or disable pasting back
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4 --no_flag_pasteback

# more options to see
python inference.py -h
```

**More interesting results can be found in our [Homepage](https://liveportrait.github.io/)** 😊

### 4. Gradio interface (WIP)

We also provide a Gradio interface for a better experience. Please install `gradio` and then run `app.py`:

```bash
pip install gradio==4.36.1
python app.py
```

***NOTE:*** *we are working on the Gradio interface and will be upgrading it soon.*


### 5. Inference speed evaluation 🚀🚀🚀
We have also provided a script to evaluate the inference speed of each module:

```bash
python speed.py
```

Below are the results of inferring one frame on an RTX 4090 GPU using the native PyTorch framework with `torch.compile`:

| Model                             | Parameters(M) | Model Size(MB) | Inference(ms) |
|-----------------------------------|:-------------:|:--------------:|:-------------:|
| Appearance Feature Extractor      |     0.84      |       3.3      |     0.82      |
| Motion Extractor                  |     28.12     |       108      |     0.84      |
| Spade Generator                   |     55.37     |       212      |     7.59      |
| Warping Module                    |     45.53     |       174      |     5.21      |
| Stitching and Retargeting Modules|     0.23      |       2.3      |     0.31      |

*Note: the listed values of Stitching and Retargeting Modules represent the combined parameter counts and the total sequential inference time of three MLP networks.*


## Acknowledgements
We would like to thank the contributors of [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [SPADE](https://github.com/NVlabs/SPADE), [InsightFace](https://github.com/deepinsight/insightface) repositories, for their open research and contributions.

## Citation 💖
If you find LivePortrait useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{guo2024live,
  title   = {LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control},
  author  = {Jianzhu Guo and Dingyun Zhang and Xiaoqiang Liu and Zhizhou Zhong and Yuan Zhang and Pengfei Wan and Di Zhang},
  year    = {2024},
  journal = {arXiv preprint:24xx.xxxx},
}
```
