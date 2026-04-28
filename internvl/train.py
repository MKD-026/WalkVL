#!/usr/bin/env python3
"""
LoRA fine-tune InternVL2-1B or InternVL2-4B on Bench2Drive-mini.

Usage:
  # RGB-only, 1B
  python train.py --model OpenGVLab/InternVL2-1B --mode rgb \
      --train data/train_rgb.jsonl --val data/val_rgb.jsonl \
      --output checkpoints/internvl2-1b-rgb

  # Multimodal, 4B
  python train.py --model OpenGVLab/InternVL2-4B --mode multimodal \
      --train data/train_multimodal.jsonl --val data/val_multimodal.jsonl \
      --output checkpoints/internvl2-4b-mm
"""

import json, argparse, os, math
from pathlib import Path

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_START_TOKEN   = '<img>'
IMG_END_TOKEN     = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IGNORE_INDEX      = -100
SYSTEM_PROMPT     = 'You are a Pedestrian-Centric Navigation AI assistant.'


# ── Image preprocessing (mirrors inference pipeline) ─────────────────────────

def build_transform(size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(ar, ratios, w, h, size):
    best_diff, best = float('inf'), (1, 1)
    area = w * h
    for r in ratios:
        diff = abs(ar - r[0] / r[1])
        if diff < best_diff or (diff == best_diff and area > 0.5 * size ** 2 * r[0] * r[1]):
            best_diff, best = diff, r
    return best


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    ratio = find_closest_aspect_ratio(w / h, ratios, w, h, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    img = image.resize((tw, th))
    tiles = []
    for i in range(ratio[0] * ratio[1]):
        x0 = (i % ratio[0]) * image_size
        y0 = (i // ratio[0]) * image_size
        tiles.append(img.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


# ── Dataset ───────────────────────────────────────────────────────────────────

class PedNavDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, num_image_token: int,
                 max_length: int = 2048, max_num_tiles: int = 12, image_size: int = 448):
        self.records         = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer       = tokenizer
        self.num_image_token = num_image_token
        self.max_length      = max_length
        self.max_num_tiles   = max_num_tiles
        self.transform       = build_transform(image_size)
        self.ctx_id          = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        image = Image.open(rec['image']).convert('RGB')
        tiles = dynamic_preprocess(image, max_num=self.max_num_tiles, use_thumbnail=True)
        pixel_values = torch.stack([self.transform(t) for t in tiles])  # [N, 3, H, W]
        num_patches  = len(tiles)

        human_val = rec['conversations'][0]['value']   # contains '<image>'
        answer    = rec['conversations'][1]['value']

        # Replace <image> with InternVL2 image tokens
        img_tokens = (IMG_START_TOKEN
                      + IMG_CONTEXT_TOKEN * (num_patches * self.num_image_token)
                      + IMG_END_TOKEN)
        human_val = human_val.replace('<image>', img_tokens, 1)

        # Build full conversation using tokenizer chat template
        messages = [
            {'role': 'system',    'content': SYSTEM_PROMPT},
            {'role': 'user',      'content': human_val},
            {'role': 'assistant', 'content': answer},
        ]
        # Prompt-only version to find where assistant tokens start
        prompt_messages = messages[:-1]

        full_text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

        full_ids   = self.tokenizer(full_text,   add_special_tokens=False).input_ids
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

        # Truncate
        full_ids = full_ids[:self.max_length]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels    = input_ids.clone()
        # Mask all tokens that belong to system+user (i.e., the prompt)
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = IGNORE_INDEX

        return {
            'pixel_values':   pixel_values,
            'input_ids':      input_ids,
            'attention_mask': torch.ones_like(input_ids),
            'labels':         labels,
            'image_flags':    torch.ones(num_patches, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad sequences and cat tiles across samples."""
    max_len = max(b['input_ids'].size(0) for b in batch)

    input_ids      = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b['input_ids'].size(0)
        input_ids[i, :L]      = b['input_ids']
        attention_mask[i, :L] = b['attention_mask']
        labels[i, :L]         = b['labels']

    pixel_values = torch.cat([b['pixel_values'] for b in batch], dim=0)
    image_flags  = torch.cat([b['image_flags']  for b in batch], dim=0)

    return {
        'pixel_values':   pixel_values,
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
        'image_flags':    image_flags,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'Device: {device}  dtype: {dtype}')

    # ── Load model & tokenizer ───────────────────────────────────────────────
    print(f'Loading {args.model} ...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False)

    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=not args.no_flash_attn,
    ).to(device)

    # Model's forward() expects this to be set (normally only set in .chat())
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print(f'img_context_token_id = {model.img_context_token_id}')

    # ── Freeze vision encoder, keep projection (mlp1) trainable ─────────────
    for name, param in model.named_parameters():
        param.requires_grad = False          # freeze everything first
    for name, param in model.named_parameters():
        if 'mlp1' in name:                   # vision→language projection
            param.requires_grad = True

    # ── Apply LoRA to language model ─────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.print_trainable_parameters()

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_name = f'{args.model.split("/")[-1]}-{args.mode}'
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            'model':       args.model,
            'mode':        args.mode,
            'epochs':      args.epochs,
            'lr':          args.lr,
            'lora_r':      args.lora_r,
            'grad_accum':  args.grad_accum,
            'batch_size':  args.batch_size,
            'max_length':  args.max_length,
            'save_every':  args.save_every,
        },
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    num_image_token = model.num_image_token
    print(f'num_image_token = {num_image_token}')

    train_ds = PedNavDataset(args.train, tokenizer, num_image_token,
                             max_length=args.max_length)
    val_ds   = PedNavDataset(args.val,   tokenizer, num_image_token,
                             max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    # Local metrics log (JSONL) — one line per event, easy to load with pandas
    metrics_path = out_dir / 'metrics.jsonl'
    metrics_f    = open(metrics_path, 'a')
    def log_metric(**kw):
        metrics_f.write(json.dumps(kw) + '\n')
        metrics_f.flush()

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            pv  = batch['pixel_values'].to(device, dtype=dtype)
            ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            lbl  = batch['labels'].to(device)
            flags = batch['image_flags'].to(device)

            out = model(
                pixel_values=pv,
                input_ids=ids,
                attention_mask=attn,
                labels=lbl,
                image_flags=flags,
            )
            loss = out.loss / args.grad_accum
            loss.backward()
            running_loss += out.loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                cur_lr = scheduler.get_last_lr()[0]
                wandb.log({'train/loss': out.loss.item(), 'train/lr': cur_lr})
                log_metric(event='train_step', epoch=epoch, step=global_step,
                           train_loss=out.loss.item(), lr=cur_lr)

            if (step + 1) % 50 == 0:
                avg = running_loss / 50
                print(f'  Epoch {epoch}  step {step+1}/{len(train_loader)}  loss={avg:.4f}')
                running_loss = 0.0

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pv   = batch['pixel_values'].to(device, dtype=dtype)
                ids  = batch['input_ids'].to(device)
                attn = batch['attention_mask'].to(device)
                lbl  = batch['labels'].to(device)
                flags = batch['image_flags'].to(device)
                out  = model(pixel_values=pv, input_ids=ids, attention_mask=attn,
                             labels=lbl, image_flags=flags)
                val_loss  += out.loss.item()
                val_steps += 1

        val_loss /= max(val_steps, 1)
        print(f'Epoch {epoch}  val_loss={val_loss:.4f}')
        wandb.log({'val/loss': val_loss, 'epoch': epoch})
        log_metric(event='val_epoch', epoch=epoch, step=global_step, val_loss=val_loss)

        # Save every N epochs
        if epoch % args.save_every == 0:
            ckpt_dir = out_dir / f'lora_epoch{epoch:02d}'
            model.language_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f'  Saved checkpoint → {ckpt_dir}')

        # Save best checkpoint (LoRA weights only)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.language_model.save_pretrained(out_dir / 'lora_best')
            tokenizer.save_pretrained(out_dir / 'lora_best')
            wandb.log({'val/best_loss': best_val_loss, 'epoch': epoch})
            log_metric(event='best_checkpoint', epoch=epoch, val_loss=val_loss)
            print(f'  Saved best checkpoint (val_loss={val_loss:.4f})')

    # Save final checkpoint
    model.language_model.save_pretrained(out_dir / 'lora_final')
    tokenizer.save_pretrained(out_dir / 'lora_final')
    print(f'Training complete. Best val_loss={best_val_loss:.4f}')
    metrics_f.close()
    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         default='OpenGVLab/InternVL2-1B')
    parser.add_argument('--mode',          choices=['rgb', 'multimodal'], required=True)
    parser.add_argument('--train',         required=True, help='Path to train JSONL')
    parser.add_argument('--val',           required=True, help='Path to val JSONL')
    parser.add_argument('--output',        required=True, help='Checkpoint output dir')
    parser.add_argument('--epochs',        type=int,   default=40)
    parser.add_argument('--save-every',    type=int,   default=5,
                        help='Save a LoRA checkpoint every N epochs')
    parser.add_argument('--batch-size',    type=int,   default=1)
    parser.add_argument('--grad-accum',    type=int,   default=8,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    parser.add_argument('--lr',            type=float, default=2e-4)
    parser.add_argument('--lora-r',        type=int,   default=16)
    parser.add_argument('--max-length',    type=int,   default=4096)
    parser.add_argument('--no-flash-attn',  action='store_true')
    parser.add_argument('--wandb-project',  default='internvl2-ped-nav',
                        help='WandB project name')
    args = parser.parse_args()
    args.save_every = args.save_every  # passed through args namespace
    train(args)


if __name__ == '__main__':
    main()
