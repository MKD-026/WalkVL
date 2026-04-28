#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from packaging import version
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def colorize_segmentation_mask(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask)
    if mask_np.ndim == 3 and mask_np.shape[2] >= 3:
        return Image.fromarray(mask_np[:, :, :3].astype(np.uint8), mode="RGB")
    if mask_np.ndim == 3 and mask_np.shape[2] == 1:
        mask_np = mask_np[:, :, 0]
    if mask_np.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

    labels = mask_np.astype(np.int64)
    red = (labels * 37) % 255
    green = (labels * 67 + 29) % 255
    blue = (labels * 97 + 101) % 255
    colored = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    colored[labels == 0] = 0
    return Image.fromarray(colored, mode="RGB")


def build_segmentation_inputs(
    rgb_path: Path, mask_path: Path, seg_mode: str, mask_alpha: float
):
    rgb = Image.open(rgb_path).convert("RGB")
    mask_raw = Image.open(mask_path)
    if mask_raw.size != rgb.size:
        mask_raw = mask_raw.resize(rgb.size, resample=Image.Resampling.NEAREST)
    mask_rgb = colorize_segmentation_mask(mask_raw)
    overlay = Image.blend(rgb, mask_rgb, alpha=mask_alpha)

    if seg_mode == "overlay":
        return [overlay], ["RGB image with segmentation overlay"]
    if seg_mode == "separate":
        return [rgb, mask_rgb], ["RGB image", "Segmentation mask (pseudo-color)"]
    return [rgb, mask_rgb, overlay], [
        "RGB image",
        "Segmentation mask (pseudo-color)",
        "RGB image with segmentation overlay",
    ]


def load_image(image_input, input_size=448, max_num=12):
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def load_question_from_json(json_path: Path, key: str = "question") -> str:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path} must contain a JSON object.")
    if key not in payload:
        raise KeyError(
            f"Key '{key}' not found in {json_path}. Available keys: {sorted(payload.keys())}"
        )
    question = payload[key]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"JSON key '{key}' in {json_path} must be a non-empty string.")
    return question


def load_model(model_id: str, dtype, use_flash_attn: bool, load_in_8bit: bool):
    dtype_key = "dtype" if version.parse(transformers_version).major >= 5 else "torch_dtype"
    kwargs = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
    )
    kwargs[dtype_key] = dtype
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"

    try:
        model = AutoModel.from_pretrained(model_id, **kwargs).eval()
    except Exception as exc:
        message = str(exc)
        if "all_tied_weights_keys" in message:
            if kwargs.get("low_cpu_mem_usage", False):
                print(
                    "Detected transformers/InternVL loading incompatibility. "
                    "Retrying with low_cpu_mem_usage=False...",
                    file=sys.stderr,
                )
                retry_kwargs = dict(kwargs)
                retry_kwargs["low_cpu_mem_usage"] = False
                try:
                    model = AutoModel.from_pretrained(model_id, **retry_kwargs).eval()
                    if not load_in_8bit:
                        model = model.cuda()
                    return model
                except Exception as retry_exc:
                    raise RuntimeError(
                        "InternVL2-8B is incompatible with your installed transformers version "
                        f"({transformers_version}). Downgrade to a 4.x release, e.g.:\n"
                        'python -m pip install --upgrade --force-reinstall "transformers==4.49.0" "huggingface_hub<1.0"'
                    ) from retry_exc
            raise RuntimeError(
                "InternVL2-8B is incompatible with your installed transformers version "
                f"({transformers_version}). Downgrade to a 4.x release, e.g.:\n"
                'python -m pip install --upgrade --force-reinstall "transformers==4.49.0" "huggingface_hub<1.0"'
            ) from exc
        if use_flash_attn and "flash" in str(exc).lower():
            print(
                "flash-attn not available in this environment, retrying with use_flash_attn=False",
                file=sys.stderr,
            )
            kwargs["use_flash_attn"] = False
            model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        else:
            raise

    if not load_in_8bit:
        model = model.cuda()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenGVLab/InternVL2-8B on RGB image(s) and optional segmentation masks."
    )
    parser.add_argument("images", nargs="+", help="Path(s) to input image(s).")
    parser.add_argument(
        "--seg-mask",
        default=None,
        help="Optional segmentation mask path paired with a single RGB input image.",
    )
    parser.add_argument(
        "--seg-mode",
        choices=["separate", "overlay", "both"],
        default="separate",
        help="How to feed RGB + mask: separate images, overlay only, or both.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Overlay opacity for mask in [0,1] when seg-mode uses overlay.",
    )
    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL2-8B",
        help="Hugging Face model id (default: OpenGVLab/InternVL2-8B).",
    )
    parser.add_argument(
        "--question",
        default="Describe the image(s) in detail.",
        help="Prompt text asked to the model.",
    )
    parser.add_argument(
        "--question-json",
        default=None,
        help="Optional JSON file containing the prompt text (default key: 'question').",
    )
    parser.add_argument(
        "--question-key",
        default="question",
        help="JSON key used with --question-json (default: question).",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=12,
        help="Max number of image tiles per image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling (default is deterministic decoding).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (used only with --sample).",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention explicitly.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load with bitsandbytes 8-bit quantization to reduce VRAM.",
    )
    args = parser.parse_args()

    missing = [img for img in args.images if not Path(img).exists()]
    if args.seg_mask and not Path(args.seg_mask).exists():
        missing.append(args.seg_mask)
    if args.question_json and not Path(args.question_json).exists():
        missing.append(args.question_json)
    if missing:
        raise FileNotFoundError(f"Image path(s) not found: {missing}")
    if args.seg_mask and len(args.images) != 1:
        raise ValueError("--seg-mask currently expects exactly one RGB image path.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be in [0, 1].")
    if args.question_json:
        args.question = load_question_from_json(
            json_path=Path(args.question_json), key=args.question_key
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. Start an interactive GPU session/job first."
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype={dtype}, cuda_device={torch.cuda.get_device_name(0)}", file=sys.stderr)

    model = load_model(
        model_id=args.model,
        dtype=dtype,
        use_flash_attn=not args.no_flash_attn,
        load_in_8bit=args.load_in_8bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False
    )

    if args.seg_mask:
        model_images, image_roles = build_segmentation_inputs(
            rgb_path=Path(args.images[0]),
            mask_path=Path(args.seg_mask),
            seg_mode=args.seg_mode,
            mask_alpha=args.mask_alpha,
        )
    else:
        model_images = [Path(image_path) for image_path in args.images]
        image_roles = []

    batches = []
    num_patches_list = []
    for image_input in model_images:
        pixel_values = load_image(image_input, max_num=args.max_num)
        num_patches_list.append(pixel_values.size(0))
        batches.append(pixel_values.to(dtype).cuda())
    pixel_values = torch.cat(batches, dim=0)

    if len(model_images) == 1:
        question = (
            args.question
            if "<image>" in args.question
            else f"<image>\n{args.question}"
        )
        extra_kwargs = {}
    else:
        if image_roles:
            image_tags = "\n".join(
                [
                    f"Image-{idx + 1} ({role}): <image>"
                    for idx, role in enumerate(image_roles)
                ]
            )
        else:
            image_tags = "\n".join(
                [f"Image-{idx + 1}: <image>" for idx in range(len(model_images))]
            )
        question = f"{image_tags}\n{args.question}"
        extra_kwargs = {"num_patches_list": num_patches_list}

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
    }
    if args.sample:
        generation_config["temperature"] = args.temperature

    output = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        **extra_kwargs,
    )
    response = output[0] if isinstance(output, tuple) else output
    print(response)


if __name__ == "__main__":
    main()
