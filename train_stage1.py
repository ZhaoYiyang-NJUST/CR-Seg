#!/usr/bin/env python3
"""
CORE-Seg Stage-Aware V4 Stage 1: SINGLE-GPU SFT (Per-Stage Seg Tokens)
========================================================================

V4 SINGLE-GPU LAYOUT (1× RTX 4090 24GB):
  Eff batch = 1 × grad_accum_steps (default 9, matches DDP 1×3×3=9)
  Peak memory ~20GB/24GB ✅ (with gradient checkpointing + fused AdamW)

USAGE:
  python train_stage1_stage_aware_v4.py \
      --epochs 50 --lr 5e-5 --grad_accum_steps 9 \
      --output_dir ./checkpoints/stage1_sa_v4

  # With existing data (auto-converts):
  python train_stage1_stage_aware_v4.py \
      --train_data ./training_data_progressive/train.json \
      --test_data ./training_data_progressive/test.json \
      --convert_v4_data
"""

import argparse, os, math, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import numpy as np
from PIL import Image as PILImage


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser(description="Stage-Aware V4 Stage 1 Single-GPU SFT")
    ap.add_argument("--mllm_path", default="/opt/data/private/models/Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--sam_path", default="/opt/data/private/ZYY/paper3/MedSAM2-main")
    ap.add_argument("--sam_ckpt", default="/opt/data/private/ZYY/paper3/MedSAM2-main/checkpoints/MedSAM2_latest.pt")
    ap.add_argument("--sam_config", default="configs/sam2.1_hiera_t512")
    ap.add_argument("--sam_dtype", default="bf16", choices=["fp32","fp16","bf16"])
    ap.add_argument("--train_data", default="./training_data_progressive/train.json")
    ap.add_argument("--test_data", default="./training_data_progressive/test.json")
    ap.add_argument("--convert_v4_data", action="store_true",
                    help="Auto-convert V3 data (flat) to V4 format (per-stage seg tokens)")
    ap.add_argument("--train_data_v4", default=None,
                    help="V4-specific training data path (传了就直接用该数据训练)")
    ap.add_argument("--test_data_v4", default=None,
                    help="V4-specific test data path (传了就直接用该数据训练)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=9,
                    help="Gradient accumulation steps (eff batch = 1 × grad_accum)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lambda_text", type=float, default=1.0)
    ap.add_argument("--lambda_seg", type=float, default=1.0)
    ap.add_argument("--lambda_cf", type=float, default=0.05)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--adapter_tokens_per_stage", type=int, default=2)
    ap.add_argument("--adapter_output_tokens", type=int, default=6)
    ap.add_argument("--adapter_resmlp_layers", type=int, default=1)
    ap.add_argument("--no_counterfactual", action="store_true")
    ap.add_argument("--output_dir", default="./checkpoints/stage1_sa_v4_nonreentrant")
    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--eval_interval", type=int, default=5,
                    help="每 N 个 epoch 测试一次 (默认 5)")
    ap.add_argument("--eval_start_epoch", type=int, default=20,
                    help="从第 N 个 epoch 才开始测试 (默认 20, 前 N-1 个 epoch 不测)")
    ap.add_argument("--eval_samples", type=int, default=0, help="Eval samples (0=full test set)")
    ap.add_argument("--seed", type=int, default=-1,
                    help="随机种子 (-1=不固定, 默认; 传入 >=0 则固定)")
    ap.add_argument("--resume", type=str, default="",
                    help="从指定 checkpoint 目录恢复训练 (如 ./checkpoints/stage1_sa_v5/epoch_20)")
    return ap.parse_args()


def train_one_epoch(model, optimizer, scheduler, dataloader, args, device):
    model.train()
    tl = tt = ts = tc = 0.0; optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Train")
    for step, batch in enumerate(pbar):
        iids = batch["input_ids"].to(device, non_blocking=True)
        am = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        simgs = batch["sam_images"].to(device, non_blocking=True)
        gms = batch["gt_masks"].to(device, non_blocking=True)
        pv = batch.get("pixel_values")
        if pv is not None: pv = pv.to(device, non_blocking=True)
        mkw = {}
        if "image_grid_thw" in batch:
            mkw["image_grid_thw"] = batch["image_grid_thw"].to(device, non_blocking=True)

        out = model.forward_stage1(pixel_values=pv, input_ids=iids, attention_mask=am,
                                    labels=labels, sam_images=simgs, gt_masks=gms,
                                    lambda_cf=args.lambda_cf, **mkw)
        loss = (args.lambda_text*out["text_loss"] + args.lambda_seg*out["seg_loss"] +
                args.lambda_cf*out["cf_loss"]) / args.grad_accum_steps
        loss.backward()
        if (step+1) % args.grad_accum_steps == 0 or (step+1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        tl += loss.item()*args.grad_accum_steps; tt += out["text_loss"].item()
        ts += out["seg_loss"].item(); tc += out["cf_loss"].item()
        del out, loss
        if (step+1) % 50 == 0: torch.cuda.empty_cache()
        if (step+1) % args.log_interval == 0:
            n = step+1
            pbar.set_postfix({"loss":f"{tl/n:.4f}","text":f"{tt/n:.4f}",
                              "seg":f"{ts/n:.4f}","cf":f"{tc/n:.4f}"})


@torch.no_grad()
def evaluate(model, test_dataset, processor, device, max_samples=None):
    from data.dataset import _build_inference_messages_v4, _build_training_messages_v4
    model.eval()
    ar_dice = 0.0; tf_dice = 0.0; cnt = 0
    indices = list(range(len(test_dataset)))
    if max_samples and max_samples>0 and max_samples<len(indices):
        import random; rng=random.Random(42); indices=rng.sample(indices,max_samples)

    pbar = tqdm(indices, desc="Eval")
    for i in pbar:
        item = test_dataset.data[i]; img = PILImage.open(item["image_path"]).convert("RGB")
        gt_mask = np.array(PILImage.open(item["mask_path"]).convert("L").resize(
            (model.sam_image_size,model.sam_image_size),PILImage.NEAREST)).astype(np.float32)/255.0

        # TF Dice: use GT target_text with V4 prompt
        target = item.get("target_text", "")
        if target:
            msgs_tf = _build_training_messages_v4(img, item["query"], target)
            tt = processor.apply_chat_template(msgs_tf, tokenize=False, add_generation_prompt=False)
            ins_tf = processor(text=[tt], images=[img], padding=True, return_tensors="pt").to(device)
            mkw_tf = {}
            if "image_grid_thw" in ins_tf: mkw_tf["image_grid_thw"] = ins_tf["image_grid_thw"]
            out_tf = model.mllm(input_ids=ins_tf.input_ids, attention_mask=ins_tf.attention_mask,
                                pixel_values=ins_tf.get("pixel_values"),
                                output_hidden_states=True, return_dict=True, **mkw_tf)
            hg,hc,hf = model.extract_stage_hiddens(ins_tf.input_ids, out_tf.hidden_states[-1])
            probes,film,_ = model.adapter(hg,hc,hf)
            si = img.resize((model.sam_image_size, model.sam_image_size))
            st = torch.from_numpy(np.array(si).astype(np.float32).transpose(2,0,1)/255.0).unsqueeze(0)
            pl_tf = model.decode_mask_sam(model.encode_image_sam(st), probes, film)
            pred_tf = (torch.sigmoid(pl_tf[0,0].float()).cpu().numpy()>0.5).astype(np.float32)
            inter = (pred_tf*gt_mask).sum(); uv = pred_tf.sum()+gt_mask.sum()
            tf_dice += float(2*inter/uv if uv>0 else 1.0)

        # AR Dice with V4 prompt
        msgs = _build_inference_messages_v4(img, item["query"])
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ins = processor(text=[txt], images=[img], padding=True, return_tensors="pt").to(device)
        gids = model.mllm.generate(**ins, max_new_tokens=512, do_sample=False)
        gent = processor.batch_decode(gids[:,ins.input_ids.shape[1]:], skip_special_tokens=False)[0]
        msgs_f = _build_training_messages_v4(img, item["query"], gent)
        tf = processor.apply_chat_template(msgs_f, tokenize=False, add_generation_prompt=False)
        ins_f = processor(text=[tf], images=[img], padding=True, return_tensors="pt").to(device)
        mkw = {}
        if "image_grid_thw" in ins_f: mkw["image_grid_thw"] = ins_f["image_grid_thw"]
        out = model.mllm(input_ids=ins_f.input_ids, attention_mask=ins_f.attention_mask,
                         pixel_values=ins_f.get("pixel_values"),
                         output_hidden_states=True, return_dict=True, **mkw)
        hg,hc,hf = model.extract_stage_hiddens(ins_f.input_ids, out.hidden_states[-1])
        probes,film,_ = model.adapter(hg,hc,hf)
        si = img.resize((model.sam_image_size, model.sam_image_size))
        st = torch.from_numpy(np.array(si).astype(np.float32).transpose(2,0,1)/255.0).unsqueeze(0)
        pl = model.decode_mask_sam(model.encode_image_sam(st), probes, film)
        pred = (torch.sigmoid(pl[0,0].float()).cpu().numpy()>0.5).astype(np.float32)
        inter = (pred*gt_mask).sum(); uv = pred.sum()+gt_mask.sum()
        ar_dice += float(2*inter/uv if uv>0 else 1.0)
        cnt += 1
        if cnt%10==0: pbar.set_postfix({"TF":f"{tf_dice/cnt:.4f}","AR":f"{ar_dice/cnt:.4f}"})
    return tf_dice/max(cnt,1), ar_dice/max(cnt,1)


def save_ckpt(model, save_dir, extra=None):
    os.makedirs(save_dir, exist_ok=True)
    model.mllm.save_pretrained(save_dir)
    # ── 保存输入/输出 embedding（special token 的 embedding 是训练的，必须持久化）──
    in_emb = model.mllm.get_input_embeddings()
    torch.save(in_emb.weight.detach().cpu(), os.path.join(save_dir, "embed_tokens.pth"))
    out_emb = model.mllm.get_output_embeddings()
    if out_emb is not None and out_emb.weight is not in_emb.weight:
        torch.save(out_emb.weight.detach().cpu(), os.path.join(save_dir, "lm_head.pth"))
    torch.save(model.adapter.state_dict(), os.path.join(save_dir,"adapter.pth"))
    torch.save(model.sam_mask_decoder.state_dict(), os.path.join(save_dir,"sam_mask_decoder.pth"))
    if model.cf_heads is not None:
        torch.save(model.cf_heads.state_dict(), os.path.join(save_dir,"cf_heads.pth"))
    if extra: torch.save(extra, os.path.join(save_dir,"training_state.pt"))
    print(f"  Saved → {save_dir}")


def convert_to_v4_format(data):
    """
    Convert V3 flat CoT to V4 per-stage seg token format.

    V3: <stage_X>...text...</stage_X>
    V4: <stage_X>...text... The mask is <seg_X>.</stage_X>

    Also updates the answer block to keep backwards compat <seg> token.
    """
    stage_tags = {
        "stage_global": "seg_global",
        "stage_contrast": "seg_contrast",
        "stage_focus": "seg_focus",
    }
    text = data
    for stage_tag, seg_tag in stage_tags.items():
        # Insert " The mask is <seg_X>." before </stage_X>
        closing = f"</{stage_tag}>"
        insertion = f" The mask is <{seg_tag}>."
        text = text.replace(closing, insertion + closing)
    return text


def main():
    args = parse_args()

    if args.seed >= 0:
        set_seed(args.seed)
        print(f"  Seed: {args.seed} (固定)")

    # ── 数据选择逻辑（修复 --train_data_v4 静默忽略 bug）──
    # 优先级:
    #   1. --convert_v4_data: 把 V3 转成 V4, 然后退出
    #   2. --train_data_v4/--test_data_v4 显式传入: 直接用 V4 数据训练
    #   3. 否则: 用默认 --train_data/--test_data
    train_data_path = args.train_data
    test_data_path = args.test_data

    _DEFAULT_TRAIN_V4 = "./training_data_progressive_v4/train.json"
    _DEFAULT_TEST_V4 = "./training_data_progressive_v4/test.json"

    # Auto-convert data if requested
    if args.convert_v4_data:
        import json
        os.makedirs("./training_data_progressive_v4", exist_ok=True)
        dst_train = args.train_data_v4 or _DEFAULT_TRAIN_V4
        dst_test = args.test_data_v4 or _DEFAULT_TEST_V4
        for src_path, dst_path in [(args.train_data, dst_train),
                                    (args.test_data, dst_test)]:
            with open(src_path) as f:
                data = json.load(f)
            for item in data:
                item["target_text"] = convert_to_v4_format(item.get("target_text", ""))
            with open(dst_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[V4 Data] Converted {len(data)} samples: {src_path} → {dst_path}")
        print("[V4 Data] Conversion done. Exiting. Run again without --convert_v4_data to train.")
        import sys; sys.exit(0)

    # 显式传了 V4 数据路径 → 直接用（修复静默忽略）
    if args.train_data_v4 is not None:
        train_data_path = args.train_data_v4
    if args.test_data_v4 is not None:
        test_data_path = args.test_data_v4

    print(f"[Data] 训练数据: {train_data_path}")
    print(f"[Data] 测试数据: {test_data_path}")

    device = torch.device("cuda:0")
    eff = args.batch_size * args.grad_accum_steps
    print("="*65)
    print(f"  Stage-Aware V4 Stage 1: Single-GPU SFT (Per-Stage Seg Tokens)")
    print("="*65)
    print(f"  Eff batch: 1×{args.grad_accum_steps}={eff}")
    print(f"  LR: {args.lr} | Epochs: {args.epochs}")
    print(f"  GPU: single (24GB)")

    from model.core_seg_stage_aware_v4 import CORESegStageAwareV4
    model = CORESegStageAwareV4(
        mllm_path=args.mllm_path, sam_path=args.sam_path, sam_ckpt=args.sam_ckpt,
        sam_config=args.sam_config, sam_dtype=args.sam_dtype,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        adapter_tokens_per_stage=args.adapter_tokens_per_stage,
        adapter_output_tokens=args.adapter_output_tokens,
        adapter_resmlp_layers=args.adapter_resmlp_layers,
        use_counterfactual=not args.no_counterfactual)

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")

    embed = model.mllm.get_input_embeddings(); embed.weight.requires_grad = True
    # [NON-REENTRANT 实验] 用 non-reentrant checkpointing（use_reentrant=False）
    # 182800(0.85) 没有 "requires_grad" 警告 = 它没走 reentrant；后面 0.76 的 run 都是 reentrant 默认
    model.mllm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = model.to(device)
    print(f"  Single-GPU (No DDP)")

    from data.dataset import CoreSegDataset, collate_fn_v4
    train_ds = CoreSegDataset(data_path=train_data_path, processor=model.processor,
                              sam_image_size=model.sam_image_size)
    test_ds = CoreSegDataset(data_path=test_data_path, processor=model.processor,
                             sam_image_size=model.sam_image_size)
    print(f"  Data: {len(train_ds.data)} train, {len(test_ds.data)} test")

    collate = partial(collate_fn_v4, processor=model.processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate,
                              num_workers=2, pin_memory=True)

    tps = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(tps, lr=args.lr, weight_decay=0.01, fused=True)
    print("  fused AdamW: ENABLED")

    spe = len(train_loader)//args.grad_accum_steps
    if len(train_loader)%args.grad_accum_steps: spe += 1
    total_steps = spe*args.epochs
    ws = max(1, int(total_steps*args.warmup_ratio))
    def lr_fn(s):
        if s<ws: return float(s)/max(1,ws)
        p = float(s-ws)/max(1,total_steps-ws)
        return max(0.0, 0.5*(1.0+math.cos(math.pi*p)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    os.makedirs(args.output_dir, exist_ok=True)
    best_dice = 0.0
    start_epoch = 0

    # ── Resume 支持 ──
    if args.resume:
        resume_dir = args.resume
        print(f"\n[Resume] 从 {resume_dir} 恢复训练...")

        # 1. 加载 LoRA
        if os.path.exists(os.path.join(resume_dir, "adapter_config.json")):
            model.mllm.load_adapter(resume_dir, adapter_name="default")
            print("  [Resume] LoRA 已加载")

        # 1.5 加载 embedding（special token embedding，修复未持久化 bug）
        emb_path = os.path.join(resume_dir, "embed_tokens.pth")
        if os.path.exists(emb_path):
            model.mllm.get_input_embeddings().weight.data.copy_(
                torch.load(emb_path, map_location="cpu"))
            lm_path = os.path.join(resume_dir, "lm_head.pth")
            if os.path.exists(lm_path):
                model.mllm.get_output_embeddings().weight.data.copy_(
                    torch.load(lm_path, map_location="cpu"))
            print("  [Resume] embedding 已加载")
        else:
            print("  [Resume] ⚠️ 未找到 embed_tokens.pth，special token embedding 将保持随机（旧 checkpoint）")

        # 2. 加载 adapter / SAM decoder / CF heads
        for fname, attr, desc in [("adapter.pth", "adapter", "Adapter"),
                                   ("sam_mask_decoder.pth", "sam_mask_decoder", "SAM dec"),
                                   ("cf_heads.pth", "cf_heads", "CF heads")]:
            fp = os.path.join(resume_dir, fname)
            if os.path.exists(fp):
                if fname == "cf_heads.pth" and model.cf_heads is None:
                    continue
                st = torch.load(fp, map_location="cpu")
                getattr(model, attr).load_state_dict(st)
                print(f"  [Resume] {desc} 已加载")

        # 3. 加载 optimizer / scheduler / epoch / best_dice
        state_path = os.path.join(resume_dir, "training_state.pt")
        if os.path.exists(state_path):
            st = torch.load(state_path, map_location="cpu")
            try:
                optimizer.load_state_dict(st["optimizer"])
                scheduler.load_state_dict(st["scheduler"])
                print("  [Resume] optimizer + scheduler 已加载")
            except Exception as e:
                print(f"  [Resume] optimizer 加载失败 ({e})，从零开始")
            start_epoch = st.get("epoch", 0)
            if st.get("best_metric", "") == "ar_dice":
                best_dice = st.get("best_dice", 0.0)
            else:
                # 旧 checkpoint 的 best_dice 是 TF Dice；新代码按 AR 选，重置
                best_dice = 0.0
                print("  [Resume] ⚠️ 旧 checkpoint 记录的是 TF best，AR best 已重置为 0")
            print(f"  [Resume] epoch={start_epoch}, best_dice(AR)={best_dice:.4f}")

        print(f"  [Resume] 从 Epoch {start_epoch+1} 继续训练\n")

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, scheduler, train_loader, args, device)
        print(f"  Epoch {epoch+1}/{args.epochs} done")

        # 前 eval_start_epoch 个 epoch 不测试; 之后每 eval_interval 个 epoch 测一次
        if (epoch+1) >= args.eval_start_epoch and \
           ((epoch+1) % args.eval_interval == 0 or (epoch+1) == args.epochs):
            eval_n = args.eval_samples if args.eval_samples > 0 else None
            tf_dice, ar_dice = evaluate(model, test_ds, model.processor, device, max_samples=eval_n)
            print(f"\n  Epoch {epoch+1} | TF Dice: {tf_dice:.4f} | AR Dice: {ar_dice:.4f} | Best AR: {best_dice:.4f}")
            if ar_dice > best_dice:
                best_dice = ar_dice
                save_ckpt(model, os.path.join(args.output_dir,"best_model"),
                          {"epoch":epoch+1,"best_dice":best_dice,
                           "best_tf_dice":tf_dice,"best_metric":"ar_dice"})
        if (epoch+1)%10==0:
            save_ckpt(model, os.path.join(args.output_dir,f"epoch_{epoch+1}"),
                      {"epoch":epoch+1,"best_dice":best_dice,"best_metric":"ar_dice",
                       "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict()})

    save_ckpt(model, os.path.join(args.output_dir,"final_model"),
              {"epoch":args.epochs,"best_dice":best_dice,"best_metric":"ar_dice"})
    print(f"\n  Done. Best AR Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
