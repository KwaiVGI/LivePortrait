# LivePortrait for Nuke

## Introduction 📖


This project integrates  [**LivePortrait**: Efficient Portrait Animation with Stitching and Retargeting Control](https://liveportrait.github.io/) to **The Foundry's Nuke**, enabling artists to easily create animated portraits through advanced facial expression and motion transfer.

**LivePortrait** leverages a series of neural networks to extract information, deform, and blend reference videos with target images, producing highly realistic and expressive animations.

By integrating **LivePortrait** into Nuke, artists can enhance their workflows within a familiar environment, gaining additional control through Nuke's curve editor and custom knob creation.

This implementation provides a self-contained package as a series of **Inference** nodes. This allows for easy installation on any Nuke 14+ system, **without requiring additional dependencies** like ComfyUI or conda environments.

The current version supports video-to-image animation transfer. Future developments will expand this functionality to include video-to-video animation transfer, eyes and lips retargeting, an animal animation model, and support for additional face detection models.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-silva-ba166513/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

<p align="center">
  <img src="./assets/docs/showcase2.gif" alt="showcase">
  <br>
  🔥 For more results, visit the project <a href="https://liveportrait.github.io/"><strong>homepage</strong></a> 🔥
</p>


## Compatibility

**Nuke 15.1+**, tested on **Linux**.


## Features

- **Fast** inference and animation transfer
- **Flexible** advanced options for animation control
- **Seamless integration** into Nuke's node graph and curve editor
- **Separated** network nodes for **customization** and workflow experimentation
- **Easy installation** using Nuke's Cattery system


## Limitations

> Maximum resolution for image output is currently 256x256 pixels (upscaled to 512x512 pixels), due to the original model's limitations.


## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/LivePortrait-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**LivePortrait** will then be accessible under the toolbar at **Cattery > Stylization > LivePortrait**.


## Quick Start

LivePortrait requires two inputs:

- **Image** (target face)
- **Video reference** (animation to be transferred)

Open the included `demo.nk` file for a working example.
A self-contained gizmo will be provided in the next release.


## Release Notes

**Latest version:** 1.0

- [x] Initial release
- [x] Video to image animation transfer
- [x] Integrated into Nuke's node graph
- [x] Advanced options for animation control
- [x] Easy installation with Cattery package


## License and Acknowledgments

**LivePortrait.cat** is licensed under the MIT License, and is derived from https://github.com/KwaiVGI/LivePortrait.

While the MIT License permits commercial use of **LivePortrait**, the dataset used for its training and some of the underlying models may be under a non-commercial license.

This license **does not cover** the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

Consult https://github.com/KwaiVGI/LivePortrait for more information on associated licensing terms.

**Users are solely responsible for ensuring that the underlying model, training data, and dependencies align with their intended usage of LivePortrait.cat.**


## Community Resources 🤗

Discover the invaluable resources contributed by our community to enhance your LivePortrait experience:

- [ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ) by [@kijai](https://github.com/kijai)
- [comfyui-liveportrait](https://github.com/shadowcz007/comfyui-liveportrait) by [@shadowcz007](https://github.com/shadowcz007)
- [LivePortrait hands-on tutorial](https://www.youtube.com/watch?v=uyjSTAOY7yI) by [@AI Search](https://www.youtube.com/@theAIsearch)
- [ComfyUI tutorial](https://www.youtube.com/watch?v=8-IcDDmiUMM) by [@Sebastian Kamph](https://www.youtube.com/@sebastiankamph)
- [LivePortrait In ComfyUI](https://www.youtube.com/watch?v=aFcS31OWMjE) by [@Benji](https://www.youtube.com/@TheFutureThinker)
- [Replicate Playground](https://replicate.com/fofr/live-portrait) and [cog-comfyui](https://github.com/fofr/cog-comfyui) by [@fofr](https://github.com/fofr)

And many more amazing contributions from our community!

## Acknowledgements
We would like to thank the contributors of [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [SPADE](https://github.com/NVlabs/SPADE), [InsightFace](https://github.com/deepinsight/insightface) repositories, for their open research and contributions.

## Citation
If you find LivePortrait useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{guo2024liveportrait,
  title   = {LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control},
  author  = {Guo, Jianzhu and Zhang, Dingyun and Liu, Xiaoqiang and Zhong, Zhizhou and Zhang, Yuan and Wan, Pengfei and Zhang, Di},
  journal = {arXiv preprint arXiv:2407.03168},
  year    = {2024}
}
```
