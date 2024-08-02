## Install FFmpeg

Make sure you have `ffmpeg` and `ffprobe` installed on your system. If you don't have them installed, follow the instructions below.

> [!Note]
> The installation is copied from [SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 🤗

### Conda Users

```bash
conda install ffmpeg
```

### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the GPT-SoVITS root.

### MacOS Users
```bash
brew install ffmpeg
```
