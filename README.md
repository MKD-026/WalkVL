# WalkVL: Pedestrian-Centric Scene Understanding for Navigation

![WalkVL logo](assets/logo/walkvl_logo.png)

WalkVL explores pedestrian navigation with vision-language models conditioned on RGB, semantic segmentation, and depth cues. This repository includes a compact set of example inputs, segmentation/depth results, route demos, and presentation materials for the project.

## Project Materials

- [Slide deck](https://docs.google.com/presentation/d/1NGJPoBRbWxpnlnOHXVFDGby9GAbXLqCBP_rLQ6wy3f0/edit?usp=sharing)
- [Flowchart](assets/flowchart/walkvl_flowchart.png)

![WalkVL flowchart](assets/flowchart/walkvl_flowchart.png)

## Example Inputs and Results

| Type | Examples |
| --- | --- |
| Input overview | [inputs.jpg](assets/examples/inputs.jpg) |
| RGB frames | [sample_rgb_1.jpg](assets/examples/sample_rgb_1.jpg), [sample_rgb_2.jpg](assets/examples/sample_rgb_2.jpg) |
| Depth outputs | [sample_depth_1.jpg](assets/examples/sample_depth_1.jpg), [sample_depth_2.jpg](assets/examples/sample_depth_2.jpg) |
| Segmentation outputs | [segmentation_result_1.jpg](assets/examples/segmentation_result_1.jpg), [segmentation_result_2.jpg](assets/examples/segmentation_result_2.jpg) |

## Data Capture Setup

[train_00013_rgb_seg_2x2.gif](assets/data_capture/train_00013_rgb_seg_2x2.gif) shows an RGB and segmentation capture example used for the WalkVL data pipeline.

![RGB and segmentation data capture setup](assets/data_capture/train_00013_rgb_seg_2x2.gif)

## Demo Videos

Each video shows the route-level comparison view with RGB context, RGB+Seg+Depth grid output, and frame-aligned WalkVL outputs.

| Route | Demo video | Text output |
| --- | --- | --- |
| Route 1 | [route1_combined_rgb_outputs_grid_30fps.mp4](demos/routes/route1_combined_rgb_outputs_grid_30fps.mp4) | [route1_frame_outputs.txt](demos/text_outputs/route1_frame_outputs.txt) |
| Route 4 | [Google Drive video](https://drive.google.com/file/d/10qfLX5ABsEWY8tXPCAmmHromHz329kq5/view?usp=sharing) | [route4_frame_outputs.txt](demos/text_outputs/route4_frame_outputs.txt) |
| Route 5 | [Google Drive video](https://drive.google.com/file/d/1YdLDH38NVnIanISUazNOmQocFRKMcoo6/view?usp=sharing) | [route5_frame_outputs.txt](demos/text_outputs/route5_frame_outputs.txt) |

## Repository Layout

```text
assets/
  data_capture/  RGB and segmentation capture setup GIF
  examples/      Input, RGB, segmentation, and depth examples
  flowchart/     WalkVL project flowchart
  logo/          WalkVL logo
demos/
  routes/        Compressed local route demo videos
  text_outputs/  Frame-aligned model output summaries
internvl/
  train.py                         LoRA training entrypoint for InternVL2
  train_rgb.sh                     qsub launcher for RGB-only training
  train_mm.sh                      qsub launcher for RGB+Instance+Depth training
  train_rgb_seg_depth_lora.sh      explicit LoRA launcher for RGB+Seg+Depth
  build_dataset.py                 Bench2Drive-mini JSONL dataset builder
  batch_walkvlm_filtered.py        Batch RGB vs RGB+Seg+Depth inference
  beautify_outputs.py              Creates final readable TXT reports
  prompts/                         Prompt JSON files used by training/inference
scripts/
  clone_preprocessing_repos.sh     Clone Depth Anything V2 and LightNet
experiments/internvl/
  data/         Training JSONL files and generated multimodal composites
  checkpoints/  New training checkpoints and copied metrics summaries
  logs/         qsub logs for InternVL jobs
  outputs/      Batch inference outputs, videos, JSONL, and analyses
  final_txt/    Beautified final text reports
```

## How to Run

All new InternVL training/inference work should live under `WalkVL`, not the old `/projectnb/cs585/students/mkd/740/internVL` scratch folder.

Start from the repo root:

```bash
cd /projectnb/cs585/students/mkd/740/WalkVL
```

Load/activate the InternVL environment:

```bash
module load miniconda/24.5.0
conda activate /projectnb/cs585/students/mkd/740/envs/internvl
```

### 1. Clone Preprocessing Repos

Depth and segmentation preprocessing repos are cloned under `external/`:

```bash
bash scripts/clone_preprocessing_repos.sh
```

This clones:

```text
external/Depth-Anything-V2
external/LightNet
```

Source repos:

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- MobileNetV2Plus / LightNet: https://github.com/ansleliu/LightNet.git

Manual equivalent:

```bash
mkdir -p external
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
git clone https://github.com/ansleliu/LightNet.git external/LightNet
```

### 2. Build Training JSONL Files

Build or refresh the training dataset:

```bash
python internvl/build_dataset.py
```

This writes:

```text
experiments/internvl/data/train_rgb.jsonl
experiments/internvl/data/val_rgb.jsonl
experiments/internvl/data/train_multimodal.jsonl
experiments/internvl/data/val_multimodal.jsonl
experiments/internvl/data/composites/
```

### 3. Train InternVL LoRA Models

RGB-only LoRA fine-tuning:

```bash
qsub -v MODEL=1b /projectnb/cs585/students/mkd/740/WalkVL/internvl/train_rgb.sh
```

RGB + segmentation + depth LoRA fine-tuning:

```bash
qsub -v MODEL=1b /projectnb/cs585/students/mkd/740/WalkVL/internvl/train_rgb_seg_depth_lora.sh
```

Use `MODEL=4b` for the 4B InternVL model:

```bash
qsub -v MODEL=4b /projectnb/cs585/students/mkd/740/WalkVL/internvl/train_rgb.sh
qsub -v MODEL=4b /projectnb/cs585/students/mkd/740/WalkVL/internvl/train_rgb_seg_depth_lora.sh
```

Training logs are written to:

```text
experiments/internvl/logs/
```

Checkpoints are written to:

```text
experiments/internvl/checkpoints/
```

Full final checkpoints are not included in this repository or release archive because of file size. For access to final trained checkpoints, contact `mkd@bu.edu`.

### 4. Run Batch Inference

Run RGB and RGB+Seg+Depth inference on the filtered route set:

```bash
python internvl/batch_walkvlm_filtered.py --no-flash-attn --continue-on-error
```

Outputs are written under:

```text
experiments/internvl/outputs/filtered_walkvlm/
```

Common faster test run:

```bash
python internvl/batch_walkvlm_filtered.py \
  --routes route1 \
  --max-frames 10 \
  --no-flash-attn \
  --continue-on-error
```

### 5. Create Final Beautified TXT Reports

Regenerate all readable text summaries:

```bash
python internvl/beautify_outputs.py
```

Final polished reports are written to:

```text
/projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/final_txt
```
