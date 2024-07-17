<h1 align="center"> Webcam Live Portrait</h1>

<p align="center">
  <img src="./assets/docs/showcase2.gif" alt="showcase">
  <br>
  🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
</p>

# Webcam result

https://github.com/Mrkomiljon/Webcam_Live_Portrait/assets/92161283/4e16fbc7-8c13-4415-b946-dd731ac00b6e
    
# Live_Portrait_Monitor

# You can see [github repo](https://github.com/Mrkomiljon/Live_Portrait_Monitor) in here.

https://github.com/user-attachments/assets/80020a36-6ec9-4efa-abf7-c1adbbfc6f39

https://github.com/user-attachments/assets/471c65a2-567f-4822-93af-882f0d041f18

https://github.com/user-attachments/assets/3bf96941-3a93-475b-9d47-8c70e0bd1e48

https://github.com/user-attachments/assets/23a83942-48a6-4922-8a50-b8f7ebdaa143

https://github.com/user-attachments/assets/c65006c6-bfc2-4c99-b7d0-4d8853f0c9da

## 🔥 Updates
- **`2024/07/10`**: 🔥 I released the initial version of the inference code for webcam. Continuous updates, stay tuned!


## Introduction
This repo, named **Webcam Live Portrait**, contains the official PyTorch implementation of author paper [LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://arxiv.org/pdf/2407.03168).
I am actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

## 🔥 Getting Started
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/Mrkomiljon/Webcam_Live_Portrait.git
cd Webcam_Live_Portrait

# create env using conda
conda create -n LivePortrait python==3.9.18
conda activate LivePortrait
# install dependencies with pip
pip install -r requirements.txt
```

### 2. Download pretrained weights
Download pretrained LivePortrait weights and face detection models of InsightFace from [Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib) or [Baidu Yun](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn). We have packed all weights in one directory 😊. Unzip and place them in `./pretrained_weights` ensuring the directory structure is as follows:
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

If the script runs successfully, you will get an output mp4 file named `animations/s6--d0_concat.mp4`. This file includes the following results: driving video, input image, and generated result.

<p align="center">
  <img src="https://github.com/Mrkomiljon/Webcam_Live_Portrait/assets/92161283/7c4daf41-838d-4eb8-a762-9188cd337ee6">
</p>

# Unrealtime result

https://github.com/Mrkomiljon/Webcam_Live_Portrait/assets/92161283/7c4daf41-838d-4eb8-a762-9188cd337ee6



Or, you can change the input by specifying the `-s` and `-d` arguments come from webcam:

```bash
python inference.py -s assets/examples/source/MY_photo.jpg 

# or disable pasting back
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4 --no_flag_pasteback

# more options to see
python inference.py -h
```


### 4. Gradio interface

We also provide a Gradio interface for a better experience, just run by:

```bash
python app.py
```

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
I would like to thank the contributors of [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [SPADE](https://github.com/NVlabs/SPADE), [InsightFace](https://github.com/deepinsight/insightface) repositories, for their open research and main [authors](https://github.com/KwaiVGI/LivePortrait).

