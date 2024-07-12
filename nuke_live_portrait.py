import torch
import logging
from src.utils.helper import load_model

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


model_config = {
    "model_params": {
        "appearance_feature_extractor_params": {
            "image_channel": 3,
            "block_expansion": 64,
            "num_down_blocks": 2,
            "max_features": 512,
            "reshape_channel": 32,
            "reshape_depth": 16,
            "num_resblocks": 6,
        },
        "motion_extractor_params": {"num_kp": 21, "backbone": "convnextv2_tiny"},
        "warping_module_params": {
            "num_kp": 21,
            "block_expansion": 64,
            "max_features": 512,
            "num_down_blocks": 2,
            "reshape_channel": 32,
            "estimate_occlusion_map": True,
            "dense_motion_params": {
                "block_expansion": 32,
                "max_features": 1024,
                "num_blocks": 5,
                "reshape_depth": 16,
                "compress": 4,
            },
        },
        "spade_generator_params": {
            "upscale": 2,
            "block_expansion": 64,
            "max_features": 512,
            "num_down_blocks": 2,
        },
        "stitching_retargeting_module_params": {
            "stitching": {
                "input_size": 126,
                "hidden_sizes": [128, 128, 64],
                "output_size": 65,
            },
            "lip": {
                "input_size": 65,
                "hidden_sizes": [128, 128, 64],
                "output_size": 63,
            },
            "eye": {
                "input_size": 66,
                "hidden_sizes": [256, 256, 128, 128, 64],
                "output_size": 63,
            },
        },
    }
}


def trace_appearance_feature_extractor():
    appearance_feature_extractor = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        model_config=model_config,
        device=0,
        model_type="appearance_feature_extractor",
    )

    with torch.no_grad():
        appearance_feature_extractor.eval()
        appearance_feature_extractor = torch.jit.script(appearance_feature_extractor)

    LOGGER.info("Traced appearance_feature_extractor")
    torch.jit.save(appearance_feature_extractor, "build/appearance_feature_extractor.pt")


# trace_appearance_feature_extractor()  # done


def trace_motion_extractor():
    import src.modules.convnextv2
    motion_extractor = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        model_config=model_config,
        device=0,
        model_type="motion_extractor",
    )

    with torch.no_grad():
        motion_extractor.eval()
        motion_extractor = torch.jit.script(motion_extractor)
        # model = src.modules.convnextv2.ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], nuk_kp=21)
        # model = torch.jit.script(model.downsample_layers)

    LOGGER.info("Traced motion_extractor")
    torch.jit.save(motion_extractor, "build/motion_extractor.pt")


trace_motion_extractor()


def trace_warping_module():
    warping_module = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/warping_module.pth",
        model_config=model_config,
        device=0,
        model_type="warping_module",
    )

    with torch.no_grad():
        warping_module.eval()
        warping_module = torch.jit.script(warping_module)

    LOGGER.info("Traced warping_module")
    torch.jit.save(warping_module, "build/warping_module.pt")


# trace_warping_module()  # done


def trace_spade_generator():
    spade_generator = load_model(
        ckpt_path="./pretrained_weights/liveportrait/base_models/spade_generator.pth",
        model_config=model_config,
        device=0,
        model_type="spade_generator",
    )

    with torch.no_grad():
        spade_generator.eval()
        spade_generator = torch.jit.script(spade_generator)

    LOGGER.info("Traced spade_generator")
    torch.jit.save(spade_generator, "build/spade_generator.pt")


trace_spade_generator()  # done


def trace_stitching_retargeting_module():
    stitching_retargeting_module = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        model_config=model_config,
        device=0,
        model_type="stitching_retargeting_module",
    )

    with torch.no_grad():
        stitching = stitching_retargeting_module['stitching'].eval()
        lip = stitching_retargeting_module['lip'].eval()
        eye = stitching_retargeting_module['eye'].eval()

        stitching_trace = torch.jit.script(stitching)
        lip_trace = torch.jit.script(lip)
        eye_trace = torch.jit.script(eye)

    LOGGER.info("Traced stitching_retargeting_module")
    torch.jit.save(stitching_trace, "build/stitching_retargeting_module_stitching.pt")
    torch.jit.save(lip_trace, "build/stitching_retargeting_module_lip.pt")
    torch.jit.save(eye_trace, "build/stitching_retargeting_module_eye.pt")


trace_stitching_retargeting_module()  # done

"""

from src.modules.util import SPADEResnetBlock
from torch.nn.utils.spectral_norm import spectral_norm, SpectralNorm
from torch.nn.utils.parametrizations import spectral_norm

fin=512; fout=512; norm_G='spadespectralinstance'; label_nc=256; use_se=False; dilation=1
fmiddle = min(fin, fout)
c = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
s = spectral_norm(c)
j = spectral_norm(c).eval()
torch.jit.script(s)
torch.jit.trace(s, torch.randn(1, 512, 64, 64))
m = torch.jit.trace(j, torch.randn(1, 512, 64, 64))

with torch.no_grad():
    torch.jit.script(s)


with torch.no_grad():
    torch.jit.trace(s, torch.randn(1, 512, 64, 64))



sp = SPADEResnetBlock(fin=512, fout=512, norm_G="spadespectralinstance", label_nc=256, use_se=False, dilation=1)
sp.eval()
sp_traced = torch.jit.trace(sp, (torch.randn(1, 512, 64, 64), torch.randn(1, 256, 64, 64)))


import torch
from torch import nn

with torch.no_grad():
    c1 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(1,2,3))
    torch.jit.script(c1)


c2 = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(1,2,3))
torch.jit.script(c2)

with torch.no_grad():
    c2 = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(1,2,3))
    torch.jit.script(c2)

l = nn.Linear(20, 40)
l.weight.size()

m = torch.nn.utils.spectral_norm(nn.Linear(20, 40))
m
m.weight_u.size()

---

def modify_state_dict_inplace(model):
    state_dict = model.state_dict()
    keys_to_delete = []

    for key in list(state_dict.keys()):
        value = state_dict[key]
        k = key.split(".")

        if len(k) == 3 and k[-1] == "weight" and k[-2] in ["conv_0", "conv_1"]:
            # Register new parameters
            model.register_parameter(f"{k[0]}.{k[-2]}.weight_orig", torch.nn.Parameter(value))
            model.register_parameter(f"{k[0]}.{k[-2]}.weight_u", torch.nn.Parameter(torch.zeros_like(value)))
            model.register_parameter(f"{k[0]}.{k[-2]}.weight_v", torch.nn.Parameter(torch.zeros_like(value)))

            state_dict[f"{k[0]}.{k[-2]}.weight_orig"] = value
            keys_to_delete.append(key)
            state_dict[f"{k[0]}.{k[-2]}.weight_u"] = torch.zeros_like(value)
            state_dict[f"{k[0]}.{k[-2]}.weight_v"] = torch.zeros_like(value)

    # for key in keys_to_delete:
    #     delattr(module, key)

    # Load the modified state_dict back into the model
    model.load_state_dict(state_dict)
    return model

model = SPADEDecoder(**model_params).cuda(device)


modify_state_dict_inplace(model)
model.state_dict().keys()

model._parameters[name]

model = modify_state_dict_inplace(model) 
model.state_dict().keys()

model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage), strict=True)



modify_state_dict_inplace(model.state_dict())

modified_state_dict.keys()
model.register_parameter("aaalero", torch.nn.Parameter(torch.randn(1, 2, 3)))
model.state_dict = modified_state_dict
model = SPADEDecoder(**model_params).cuda(device)
model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage), strict=True)

type(state_dict.keys())
type(modified_state_dict)

missing_keys = ["G_middle_0.conv_0.weight_orig", "G_middle_0.conv_0.weight_u", "G_middle_0.conv_0.weight_v", "G_middle_0.conv_1.weight_orig", "G_middle_0.conv_1.weight_u", "G_middle_0.conv_1.weight_v", "G_middle_1.conv_0.weight_orig", "G_middle_1.conv_0.weight_u", "G_middle_1.conv_0.weight_v", "G_middle_1.conv_1.weight_orig", "G_middle_1.conv_1.weight_u", "G_middle_1.conv_1.weight_v", "G_middle_2.conv_0.weight_orig", "G_middle_2.conv_0.weight_u", "G_middle_2.conv_0.weight_v", "G_middle_2.conv_1.weight_orig", "G_middle_2.conv_1.weight_u", "G_middle_2.conv_1.weight_v", "G_middle_3.conv_0.weight_orig", "G_middle_3.conv_0.weight_u", "G_middle_3.conv_0.weight_v", "G_middle_3.conv_1.weight_orig", "G_middle_3.conv_1.weight_u", "G_middle_3.conv_1.weight_v", "G_middle_4.conv_0.weight_orig", "G_middle_4.conv_0.weight_u", "G_middle_4.conv_0.weight_v", "G_middle_4.conv_1.weight_orig", "G_middle_4.conv_1.weight_u", "G_middle_4.conv_1.weight_v", "G_middle_5.conv_0.weight_orig", "G_middle_5.conv_0.weight_u", "G_middle_5.conv_0.weight_v", "G_middle_5.conv_1.weight_orig", "G_middle_5.conv_1.weight_u", "G_middle_5.conv_1.weight_v", "up_0.conv_0.weight_orig", "up_0.conv_0.weight_u", "up_0.conv_0.weight_v", "up_0.conv_1.weight_orig", "up_0.conv_1.weight_u", "up_0.conv_1.weight_v", "up_0.conv_s.weight_orig", "up_0.conv_s.weight_u", "up_0.conv_s.weight_v", "up_1.conv_0.weight_orig", "up_1.conv_0.weight_u", "up_1.conv_0.weight_v", "up_1.conv_1.weight_orig", "up_1.conv_1.weight_u", "up_1.conv_1.weight_v", "up_1.conv_s.weight_orig", "up_1.conv_s.weight_u", "up_1.conv_s.weight_v"]

missing_keys in list(modified_state_dict.keys())


def apply(module: Module, name: str):
    weight = module._parameters[name]

    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    # We still need to assign weight back as fn.name because all sorts of
    # things may assume that it exists, e.g., when initializing weights.
    # However, we can't directly assign as it could be an nn.Parameter and
    # gets added as a parameter. Instead, we register weight.data as a plain
    # attribute.
    setattr(module, fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_buffer(fn.name + "_v", v)

    module.register_forward_pre_hook(fn)
    module._register_state_dict_hook(SpectralNormStateDictHook(fn))
    module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
    return fn


dims = [96, 192, 384, 768]
from typing import List


import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise ValueError(f"Unsupported data_format: {self.data_format}")
    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, ' \
               f'eps={self.eps}, data_format={self.data_format}'

               
class YourModule(nn.Module):
    def __init__(self, in_chans: int, dims: List[int]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.downsample_layers:
            x = layer(x)
        return x

        

# Now try to script the entire module
model = YourModule(3, dims)
torch.jit.script(model)

"""
