### Speed

Below are the results of inferring one frame on an RTX 4090 GPU using the native PyTorch framework with `torch.compile`:

| Model                             | Parameters(M) | Model Size(MB) | Inference(ms) |
|-----------------------------------|:-------------:|:--------------:|:-------------:|
| Appearance Feature Extractor      |     0.84      |       3.3      |     0.82      |
| Motion Extractor                  |     28.12     |       108      |     0.84      |
| Spade Generator                   |     55.37     |       212      |     7.59      |
| Warping Module                    |     45.53     |       174      |     5.21      |
| Stitching and Retargeting Modules |     0.23      |       2.3      |     0.31      |

*Note: The values for the Stitching and Retargeting Modules represent the combined parameter counts and total inference time of three sequential MLP networks.*
