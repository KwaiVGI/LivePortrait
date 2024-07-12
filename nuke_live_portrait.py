import torch
from src.utils.helper import load_model

model_config =  {'model_params': {'appearance_feature_extractor_params': {'image_channel': 3, 'block_expansion': 64, 'num_down_blocks': 2, 'max_features': 512, 'reshape_channel': 32, 'reshape_depth': 16, 'num_resblocks': 6}, 'motion_extractor_params': {'num_kp': 21, 'backbone': 'convnextv2_tiny'}, 'warping_module_params': {'num_kp': 21, 'block_expansion': 64, 'max_features': 512, 'num_down_blocks': 2, 'reshape_channel': 32, 'estimate_occlusion_map': True, 'dense_motion_params': {'block_expansion': 32, 'max_features': 1024, 'num_blocks': 5, 'reshape_depth': 16, 'compress': 4}}, 'spade_generator_params': {'upscale': 2, 'block_expansion': 64, 'max_features': 512, 'num_down_blocks': 2}, 'stitching_retargeting_module_params': {'stitching': {'input_size': 126, 'hidden_sizes': [128, 128, 64], 'output_size': 65}, 'lip': {'input_size': 65, 'hidden_sizes': [128, 128, 64], 'output_size': 63}, 'eye': {'input_size': 66, 'hidden_sizes': [256, 256, 128, 128, 64], 'output_size': 63}}}}


def trace_appearance_feature_extractor():
    appearance_feature_extractor = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        model_config=model_config,
        device=0,
        model_type='appearance_feature_extractor',
    )

    with torch.no_grad():
        appearance_feature_extractor.eval()
        appearance_feature_extractor = torch.jit.script(appearance_feature_extractor)

    torch.jit.save(appearance_feature_extractor, "build/appearance_feature_extractor.pt")


# def trace_appearance_feature_extractor():


def trace_motion_extractor():
    motion_extractor = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        model_config =  {'model_params': {'appearance_feature_extractor_params': {'image_channel': 3, 'block_expansion': 64, 'num_down_blocks': 2, 'max_features': 512, 'reshape_channel': 32, 'reshape_depth': 16, 'num_resblocks': 6}, 'motion_extractor_params': {'num_kp': 21, 'backbone': 'convnextv2_tiny'}, 'warping_module_params': {'num_kp': 21, 'block_expansion': 64, 'max_features': 512, 'num_down_blocks': 2, 'reshape_channel': 32, 'estimate_occlusion_map': True, 'dense_motion_params': {'block_expansion': 32, 'max_features': 1024, 'num_blocks': 5, 'reshape_depth': 16, 'compress': 4}}, 'spade_generator_params': {'upscale': 2, 'block_expansion': 64, 'max_features': 512, 'num_down_blocks': 2}, 'stitching_retargeting_module_params': {'stitching': {'input_size': 126, 'hidden_sizes': [128, 128, 64], 'output_size': 65}, 'lip': {'input_size': 65, 'hidden_sizes': [128, 128, 64], 'output_size': 63}, 'eye': {'input_size': 66, 'hidden_sizes': [256, 256, 128, 128, 64], 'output_size': 63}}}},
        device=0,
        model_type='motion_extractor',
    )
    # print(motion_extractor)

    # with torch.no_grad():
    #     motion_extractor.eval()
    # torch.jit.script(self.motion_extractor)
    
    motion_extractor = torch.jit.script(motion_extractor)

    torch.jit.save(motion_extractor, "build/motion_extractor.pt")

# trace_motion_extractor()

def trace_warping_module():
    warping_module = load_model(
        ckpt_path="/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/base_models/warping_module.pth",
        model_config=model_config,
        device=0,
        model_type='warping_module',
    )

    with torch.no_grad():
        warping_module.eval()
        warping_module = torch.jit.script(warping_module)

    torch.jit.save(warping_module, "build/warping_module.pt")


# def trace_warping_module():


def trace_spade_generator():
    spade_generator = load_model(
        ckpt_path='./pretrained_weights/liveportrait/base_models/spade_generator.pth',
        model_config=model_config,
        device=0,
        model_type='spade_generator',
    )

    # with torch.no_grad():
    #     spade_generator.eval()
    # print(spade_generator)
    spade_generator = torch.jit.script(spade_generator)
    torch.jit.save(spade_generator, "build/spade_generator.pt")

trace_spade_generator()



def trace_stitching_retargeting_module():
    stitching_retargeting_module = load_model(
        ckpt_path='/mnt/x/1_projects/relight/LivePortrait/src/config/../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth',
        model_config=model_config,
        device=0,
        model_type='stitching_retargeting_module',
    )

    with torch.no_grad():
        stitching_retargeting_module.eval()
        stitching_retargeting_module = torch.jit.script(stitching_retargeting_module)

    torch.jit.save(stitching_retargeting_module, "build/stitching_retargeting_module.pt")


# trace_stitching_retargeting_module()

'''
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        print(f"SPADEResnetBlock: fin={fin}, fout={fout}, norm_G={norm_G}, fmiddle={fmiddle}, learned_shortcut={self.learned_shortcut}, use_se={use_se}")
        # create conv layers
        # self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        # if self.learned_shortcut:
        #     self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        # if 'spectral' in norm_G:
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation))
        # self.conv_0: SpectralNorm = SpectralNorm.apply(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation), "weight", 1, 0, 1e-12)
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation))
        # self.conv_1: SpectralNorm = SpectralNorm.apply(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation), "weight", 1, 0, 1e-12)
        # if self.learned_shortcut:
        self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))
        # self.conv_s: SpectralNorm = SpectralNorm.apply(nn.Conv2d(fin, fout, kernel_size=1, bias=False), "weight", 1, 0, 1e-12)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        # if self.learned_shortcut:
        self.norm_s = SPADE(fin, label_nc)


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








'''