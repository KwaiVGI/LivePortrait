# coding: utf-8

"""
The entrance of the gradio for animal
"""

import os
import tyro
import subprocess
import gradio as gr
import os.path as osp
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipelineAnimal
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )
# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

gradio_pipeline_animal: GradioPipelineAnimal = GradioPipelineAnimal(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)

if args.gradio_temp_dir not in (None, ''):
    os.environ["GRADIO_TEMP_DIR"] = args.gradio_temp_dir
    os.makedirs(args.gradio_temp_dir, exist_ok=True)

def gpu_wrapped_execute_video(*args, **kwargs):
    return gradio_pipeline_animal.execute_video(*args, **kwargs)


# assets
title_md = "assets/gradio/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples_i2v = [
    [osp.join(example_portrait_dir, "s41.jpg"), osp.join(example_video_dir, "d3.mp4"), True, False, False, False],
    [osp.join(example_portrait_dir, "s40.jpg"), osp.join(example_video_dir, "d6.mp4"), True, False, False, False],
    [osp.join(example_portrait_dir, "s25.jpg"), osp.join(example_video_dir, "d19.mp4"), True, False, False, False],
]
data_examples_i2v_pickle = [
    [osp.join(example_portrait_dir, "s25.jpg"), osp.join(example_video_dir, "wink.pkl"), True, False, False, False],
    [osp.join(example_portrait_dir, "s40.jpg"), osp.join(example_video_dir, "talking.pkl"), True, False, False, False],
    [osp.join(example_portrait_dir, "s41.jpg"), osp.join(example_video_dir, "aggrieved.pkl"), True, False, False, False],
]
#################### interface logic ####################

# Define components first
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
output_video_i2v = gr.Video(autoplay=False)
output_video_concat_i2v = gr.Video(autoplay=False)
output_video_i2v_gif = gr.Image(type="numpy")


with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    gr.HTML(load_description(title_md))

    gr.Markdown(load_description("assets/gradio/gradio_description_upload_animal.md"))
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="🐱 Source Animal Image"):
                source_image_input = gr.Image(type="filepath")
                gr.Examples(
                    examples=[
                        [osp.join(example_portrait_dir, "s25.jpg")],
                        [osp.join(example_portrait_dir, "s30.jpg")],
                        [osp.join(example_portrait_dir, "s31.jpg")],
                        [osp.join(example_portrait_dir, "s32.jpg")],
                        [osp.join(example_portrait_dir, "s39.jpg")],
                        [osp.join(example_portrait_dir, "s40.jpg")],
                        [osp.join(example_portrait_dir, "s41.jpg")],
                        [osp.join(example_portrait_dir, "s38.jpg")],
                        [osp.join(example_portrait_dir, "s36.jpg")],
                    ],
                    inputs=[source_image_input],
                    cache_examples=False,
                )

            with gr.Accordion(open=True, label="Cropping Options for Source Image"):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                    scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("📁 Driving Pickle") as tab_pickle:
                    with gr.Accordion(open=True, label="Driving Pickle"):
                        driving_video_pickle_input = gr.File()
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "wink.pkl")],
                                [osp.join(example_video_dir, "shy.pkl")],
                                [osp.join(example_video_dir, "aggrieved.pkl")],
                                [osp.join(example_video_dir, "open_lip.pkl")],
                                [osp.join(example_video_dir, "laugh.pkl")],
                                [osp.join(example_video_dir, "talking.pkl")],
                                [osp.join(example_video_dir, "shake_face.pkl")],
                            ],
                            inputs=[driving_video_pickle_input],
                            cache_examples=False,
                        )
                with gr.TabItem("🎞️ Driving Video") as tab_video:
                    with gr.Accordion(open=True, label="Driving Video"):
                        driving_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                # [osp.join(example_video_dir, "d0.mp4")],
                                # [osp.join(example_video_dir, "d18.mp4")],
                                [osp.join(example_video_dir, "d19.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                                [osp.join(example_video_dir, "d6.mp4")],
                                [osp.join(example_video_dir, "d3.mp4")],
                            ],
                            inputs=[driving_video_input],
                            cache_examples=False,
                        )

                    tab_selection = gr.Textbox(visible=False)
                    tab_pickle.select(lambda: "Pickle", None, tab_selection)
                    tab_video.select(lambda: "Video", None, tab_selection)
            with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                with gr.Row():
                    flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                    scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5, maximum=0.5, step=0.01)

    with gr.Row():
        with gr.Accordion(open=False, label="Animation Options"):
            with gr.Row():
                flag_stitching = gr.Checkbox(value=False, label="stitching")
                flag_remap_input = gr.Checkbox(value=False, label="paste-back")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0, step=0.02)

    gr.Markdown(load_description("assets/gradio/gradio_description_animate_clear.md"))
    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video in the cropped image space"):
                output_video_i2v.render()
        with gr.Column():
            with gr.Accordion(open=True, label="The animated gif in the cropped image space"):
                output_video_i2v_gif.render()
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video"):
                output_video_concat_i2v.render()
    with gr.Row():
        process_button_reset = gr.ClearButton([source_image_input, driving_video_input, output_video_i2v, output_video_concat_i2v, output_video_i2v_gif], value="🧹 Clear")

    with gr.Row():
        # Examples
        gr.Markdown("## You could also choose the examples below by one click ⬇️")
    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("📁 Driving Pickle") as tab_video:
                gr.Examples(
                    examples=data_examples_i2v_pickle,
                    fn=gpu_wrapped_execute_video,
                    inputs=[
                        source_image_input,
                        driving_video_pickle_input,
                        flag_do_crop_input,
                        flag_stitching,
                        flag_remap_input,
                        flag_crop_driving_video_input,
                    ],
                    outputs=[output_image, output_image_paste_back, output_video_i2v_gif],
                    examples_per_page=len(data_examples_i2v_pickle),
                    cache_examples=False,
                )
            with gr.TabItem("🎞️ Driving Video") as tab_video:
                gr.Examples(
                    examples=data_examples_i2v,
                    fn=gpu_wrapped_execute_video,
                    inputs=[
                        source_image_input,
                        driving_video_input,
                        flag_do_crop_input,
                        flag_stitching,
                        flag_remap_input,
                        flag_crop_driving_video_input,
                    ],
                    outputs=[output_image, output_image_paste_back, output_video_i2v_gif],
                    examples_per_page=len(data_examples_i2v),
                    cache_examples=False,
                )

    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            source_image_input,
            driving_video_input,
            driving_video_pickle_input,
            flag_do_crop_input,
            flag_remap_input,
            driving_multiplier,
            flag_stitching,
            flag_crop_driving_video_input,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            tab_selection,
        ],
        outputs=[output_video_i2v, output_video_concat_i2v, output_video_i2v_gif],
        show_progress=True
    )

demo.launch(
    server_port=args.server_port,
    share=args.share,
    server_name=args.server_name
)
