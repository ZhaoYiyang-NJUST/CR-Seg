
import argparse, os, sys, re, torch, torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from PIL import Image as PILImage


REWARD_WEIGHTS = {"fmt": 0.05, "bbox": 0.20, "mask": 0.68,
                  "reasoning": 0.00, "cf_mask": 0.12,
                  "stage": 0.10, "consistency": 0.05}
FORMAT_GATE_INIT = 0.30; FORMAT_GATE_FINAL = 0.50
FORMAT_GATE_WARMUP = 2400; FORMAT_GATE_BONUS = -0.5
GATE_ENABLED = True


DIFFICULTY_WEIGHT = {"easy": 0.0, "medium": 1.0, "hard": 1.5, "very_hard": 1.5}


CF_LOG_INTERVAL = 200


def get_diff_weight(difficulty, step, total_steps):

    if difficulty == "easy":
        return 0.0
    progress = min(1.0, step / max(1, total_steps))
    target = DIFFICULTY_WEIGHT.get(difficulty, 1.0)
    return 1.0 + (target - 1.0) * progress


def parse_args():
    ap = argparse.ArgumentParser(description="Stage-Aware V4 Stage 2 GRPO")
    ap.add_argument("--mllm_path", default="/opt/data/private/models/Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--sam_path", default="/opt/data/private/ZYY/paper3/MedSAM2-main")
    ap.add_argument("--sam_ckpt", default="/opt/data/private/ZYY/paper3/MedSAM2-main/checkpoints/MedSAM2_latest.pt")
    ap.add_argument("--sam_config", default="configs/sam2.1_hiera_t512")
    ap.add_argument("--stage1_ckpt", default="./checkpoints/stage1_sa_v4_real/best_model")
    ap.add_argument("--train_data", default="./training_data_progressive_v4/train.json")
    ap.add_argument("--test_data", default="./training_data_progressive_v4/test.json")
    ap.add_argument("--adapter_tokens_per_stage", type=int, default=2)
    ap.add_argument("--adapter_output_tokens", type=int, default=6)
    ap.add_argument("--adapter_resmlp_layers", type=int, default=1)
    ap.add_argument("--no_film", action="store_true")
    ap.add_argument("--no_dialogue", action="store_true")
    ap.add_argument("--flat_cot", action="store_true")
    ap.add_argument("--use_counterfactual", action="store_true", default=True)
    ap.add_argument("--no_counterfactual", action="store_true")
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--kl_coeff", type=float, default=0.5)
    ap.add_argument("--kl_anneal_to", type=float, default=-1.0)
    ap.add_argument("--reward_mode", type=str, default="full", choices=["baseline","full","no_stage"])
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--lambda_seg", type=float, default=0.3)
    ap.add_argument("--seg_top_frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--train_samples", type=int, default=0)
    ap.add_argument("--test_samples", type=int, default=0)
    ap.add_argument("--sub_batch_size", type=int, default=4)
    ap.add_argument("--batch_gen", type=int, default=4)
    ap.add_argument("--sam_dtype", type=str, default="bf16", choices=["fp32","fp16","bf16"])
    ap.add_argument("--policy_device", type=int, default=0)
    ap.add_argument("--sam_device", type=int, default=1)
    ap.add_argument("--ref_device", type=int, default=1)
    ap.add_argument("--output_dir", default="./checkpoints/stage2_sa_v4")
    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--ckpt_interval", type=int, default=600)
    ap.add_argument("--skip_s1_eval", action="store_true", default=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--entropy_halt", action="store_true")
    ap.add_argument("--entropy_halt_threshold", type=float, default=2.5)
    ap.add_argument("--format_consistency_weight", type=float, default=0.05)
    ap.add_argument("--use_cf_mask", action="store_true", default=True)
    ap.add_argument("--no_cf_mask", action="store_true")
    ap.add_argument("--cf_mask_weight", type=float, default=0.12)
    ap.add_argument("--cf_diff_weights", type=str, default="")
    return ap.parse_args()


from model.reward import format_reward_baseline, extract_bboxes_from_text
from data.dataset import _build_inference_messages_v4, _build_training_messages_v4
sys.stdout.reconfigure(line_buffering=True)



def bbox_reward(pred_bboxes, gt_bboxes):
    if not pred_bboxes and not gt_bboxes: return 1.0
    if not pred_bboxes or not gt_bboxes: return 0.0
    from scipy.optimize import linear_sum_assignment
    n_p, n_g = len(pred_bboxes), len(gt_bboxes); cost = np.zeros((n_p, n_g))
    def _iou(b1,b2):
        x1,y1=max(b1[0],b2[0]),max(b1[1],b2[1]); x2,y2=min(b1[2],b2[2]),min(b1[3],b2[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        a1=max(0,b1[2]-b1[0])*max(0,b1[3]-b1[1]); a2=max(0,b2[2]-b2[0])*max(0,b2[3]-b2[1])
        return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0.0
    for i in range(n_p):
        for j in range(n_g): cost[i,j]=_iou(pred_bboxes[i],gt_bboxes[j])
    row,col=linear_sum_assignment(1-cost)
    miou=np.mean([cost[r,c] for r,c in zip(row,col)]) if len(row)>0 else 0.0
    tp=len(row); fp=n_p-tp; fn=n_g-tp
    p=tp/(tp+fp) if (tp+fp)>0 else 0.0; r=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*p*r/(p+r) if (p+r)>0 else 0.0
    return float(miou*f1)

def mask_reward(pred_mask, gt_mask):
    pred_bin=(pred_mask>0.5).float(); inter=(pred_bin*gt_mask).sum()
    total=pred_bin.sum()+gt_mask.sum()
    dice=float(2*inter/total) if total>0 else (1.0 if gt_mask.sum()==0 else 0.0)
    if dice<0.05:
        def _m2b(m):
            if m.sum()==0: return None
            if m.dim()>2: m=m.squeeze()
            rows=m.any(dim=-1).nonzero(); cols=m.any(dim=-2).nonzero()
            if len(rows)==0 or len(cols)==0: return None
            return[int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())]
        pb,gb=_m2b(pred_bin),_m2b(gt_mask)
        if pb is None or gb is None: return -1.0
        def _giou(b1,b2):
            x1,y1=max(b1[0],b2[0]),max(b1[1],b2[1]); x2,y2=min(b1[2],b2[2]),min(b1[3],b2[3])
            inter=max(0,x2-x1)*max(0,y2-y1)
            a1=max(0,b1[2]-b1[0])*max(0,b1[3]-b1[1]); a2=max(0,b2[2]-b2[0])*max(0,b2[3]-b2[1])
            union=a1+a2-inter; iou=inter/union if union>0 else 0.0
            cx1,cy1=min(b1[0],b2[0]),min(b1[1],b2[1]); cx2,cy2=max(b1[2],b2[2]),max(b1[3],b2[3])
            c_area=max(0,cx2-cx1)*max(0,cy2-cy1)
            return iou-(c_area-union)/c_area if c_area>0 else iou
        return _giou(pb,gb)
    return dice

def reasoning_reward(text, pred_bboxes, image_size=512):
    if not text: return 0.0,{}
    tl=text.lower(); wc=len(text.split())
    hn=bool(re.search(r'\b(normal|background|mucosal|mucosa|colon|wall|lumen|fold|haustr)',tl))
    hl=bool(re.search(r'\b(shape|colou?r|texture|boundary|border|elevation|contour|morpholog|surface|smooth|rough|lobulated|sessile|pedunculated)',tl))
    hlo=bool(re.search(r'\b(left|right|center|upper|lower|central|located|position|region|area|side)',tl))
    struct=(hn+hl+hlo)/3.0*0.25
    if pred_bboxes and hlo:
        bbox=pred_bboxes[0]; cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2; spatial=0.125
        if re.search(r'\bleft\b',tl) and cx<image_size*0.4: spatial+=0.0625
        if re.search(r'\bright\b',tl) and cx>image_size*0.6: spatial+=0.0625
        if re.search(r'\b(upper|top|superior)\b',tl) and cy<image_size*0.4: spatial+=0.0625
        if re.search(r'\b(lower|bottom|inferior)\b',tl) and cy>image_size*0.6: spatial+=0.0625
        spatial=min(0.25,spatial)
    else: spatial=0.125
    content=0.0
    banned=["mask","overlay","annotation","ground truth","bounding box","bbox","coordinate","cancer","malignant","histology"]
    if not any(b in tl for b in banned): content+=0.10
    if re.search(r'\b(polyp|lesion|protrusion|elevation|mass|growth|abnormality)',tl): content+=0.08
    if not re.search(r'\b(pixel|coordinate)\b',tl): content+=0.07
    conc=0.25 if 30<=wc<=150 else (0.15 if 15<=wc<30 or 150<wc<=200 else 0.05)
    r=max(0.0,min(1.0,struct+spatial+content+conc))
    return r,{"struct":round(struct,3),"spatial":round(spatial,3),"content":round(content,3),"concise":round(conc,3),"words":wc}



def strip_stage_reasoning(text):
    def repl(m):
        tag = m.group(1)
        inner = m.group(2)
        segs = re.findall(r'<seg_\w+>', inner)
        keep = ' '.join(segs) + ('.' if segs else '')
        return f'<stage_{tag}>{keep}</stage_{tag}>'
    return re.sub(r'<stage_(global|contrast|focus)>(.*?)</stage_\1>',
                  repl, text, flags=re.DOTALL)


def _dice(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0.5).float()
    inter = (pred_bin * gt_mask).sum()
    total = pred_bin.sum() + gt_mask.sum()
    return float(2 * inter / total) if total > 0 else (1.0 if gt_mask.sum() == 0 else 0.0)


@torch.no_grad()
def counterfactual_mask_dice(model, processor, image, query, completion,
                             backbone_out, gt_mask_tensor, device):
    stripped = strip_stage_reasoning(completion)
    if stripped == completion or not re.search(r'<stage_', stripped):
        return None
    msgs = _build_training_messages_v4(image, query, stripped)
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ins = processor(text=[txt], images=[image], padding=True, return_tensors="pt").to(device)
    mkw = {}
    if "image_grid_thw" in ins:
        mkw["image_grid_thw"] = ins["image_grid_thw"]
    out = model.mllm(input_ids=ins.input_ids, attention_mask=ins.attention_mask,
                     pixel_values=ins.get("pixel_values"),
                     output_hidden_states=True, return_dict=True, **mkw)
    hg, hc, hf = model.extract_stage_hiddens(ins.input_ids, out.hidden_states[-1])
    probes, film, _ = model.adapter(hg, hc, hf)
    pl = model.decode_mask_sam(backbone_out, probes, film)
    stripped_mask = torch.sigmoid(pl[0, 0].float()).cpu()
    del out, ins
    return stripped_mask


def get_gate_threshold(step):
    if not GATE_ENABLED: return -1.0
    if step>=FORMAT_GATE_WARMUP: return FORMAT_GATE_FINAL
    return FORMAT_GATE_INIT+(FORMAT_GATE_FINAL-FORMAT_GATE_INIT)*step/FORMAT_GATE_WARMUP

def get_kl_coeff(step, total_steps, kl_start, kl_end):
    if kl_end < 0 or total_steps <= 0:
        return kl_start
    progress = min(1.0, step / total_steps)
    return kl_start + (kl_end - kl_start) * progress

def extract_stage_texts(output_text):
    stages={}
    for tag in["stage_global","stage_contrast","stage_focus"]:
        m=re.search(rf"<{tag}>(.*?)</{tag}>",output_text,re.DOTALL)
        stages[tag]=m.group(1).strip() if m else""
    if not any(stages.values()):
        m=re.search(r"<think>(.*?)</think>",output_text,re.DOTALL)
        if m: stages["stage_global"]=m.group(1).strip()
    return stages

def consistency_reward(output_text,pred_bboxes,image_size=512):
    stages=extract_stage_texts(output_text)
    terms={"left":r'\bleft\b',"right":r'\bright\b',"upper":r'\b(upper|top|superior)\b',
           "lower":r'\b(lower|bottom|inferior)\b',"central":r'\b(central|center|middle)\b'}
    if not pred_bboxes: return{"r_consistency":0.5,"detail":{}}
    bbox=pred_bboxes[0]; cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
    cr={"left":cx<image_size*0.4,"right":cx>image_size*0.6,"upper":cy<image_size*0.4,
        "lower":cy>image_size*0.6,"central":image_size*0.25<cx<image_size*0.75 and image_size*0.25<cy<image_size*0.75}
    ss={}
    for sn,text in stages.items():
        if not text: ss[sn]=0.5; continue
        claims=[t for t,p in terms.items() if re.search(p,text.lower())]
        if not claims: ss[sn]=0.5; continue
        matches=sum(1 for c in claims if c in cr and cr[c]); ss[sn]=matches/len(claims)
    vals=[v for k,v in ss.items() if stages.get(k,"")]
    return{"r_consistency":float(np.mean(vals)) if vals else 0.5,"detail":{k:round(v,3) for k,v in ss.items()}}


# ==============================================================
# GTPO token IDs — programmatically derived for V4
# ==============================================================
def _build_v4_gtpo_set(processor):
    """Build GTPO structural token ID set for V4, including per-stage seg tokens."""
    structural_strings = [
        "<|im_start|>assistant",
        "<think>", "</think>",
        "<answer>", "</answer>",
        "<seg>",
        "<seg_global>", "<seg_contrast>", "<seg_focus>",
        "<stage_global>", "</stage_global>",
        "<stage_contrast>", "</stage_contrast>",
        "<stage_focus>", "</stage_focus>",
        "[[", "]]",
    ]
    ids = set()
    for s in structural_strings:
        tok_ids = processor.tokenizer.encode(s, add_special_tokens=False)
        ids.update(tok_ids)
    return ids


# ==============================================================
# Batch Rollout Generation (V4 prompt)
# ==============================================================
@torch.no_grad()
def generate_rollouts_batched(model, processor, image, query, group_size,
                               temp, max_new, device, batch_gen):
    if batch_gen <= 0: batch_gen = group_size
    messages = _build_inference_messages_v4(image, query)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    completions = []
    for start in range(0, group_size, batch_gen):
        g = min(batch_gen, group_size - start)
        inputs = processor(text=[text] * g, images=[image] * g,
                           padding=True, return_tensors="pt").to(device)
        gen_ids = model.mllm.generate(**inputs, max_new_tokens=max_new,
                                       do_sample=True, temperature=temp, top_p=0.9)
        for j in range(g):
            completions.append(processor.batch_decode(
                gen_ids[j:j+1, inputs.input_ids.shape[1]:], skip_special_tokens=False)[0])
        del gen_ids, inputs
    return completions


# ==============================================================
# Reward forward (V4 prompt)
# ==============================================================
@torch.no_grad()
def reward_forward(model, processor, image, query, completions,
                   gt_bboxes, gt_mask_tensor, policy_device, reward_mode,
                   sub_batch_size, global_step, backbone_out,
                   use_cf_mask=True, cf_mask_weight=0.12, diff_weight=0.0):
    G = len(completions); all_rewards = []; all_details = []; fmt_gated = 0
    gate_thr = get_gate_threshold(global_step); W = REWARD_WEIGHTS
    for start in range(0, G, sub_batch_size):
        end = min(start + sub_batch_size, G)
        sub = completions[start:end]; B = len(sub)
        texts = []
        for comp in sub:
            msgs = _build_training_messages_v4(image, query, comp)
            texts.append(processor.apply_chat_template(msgs, tokenize=False,
                         add_generation_prompt=False))
        inputs = processor(text=texts, images=[image]*B, padding=True,
                           return_tensors="pt").to(policy_device)
        mkw = {}
        if "image_grid_thw" in inputs: mkw["image_grid_thw"] = inputs["image_grid_thw"]
        outputs = model.mllm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                             pixel_values=inputs.get("pixel_values"),
                             output_hidden_states=True, return_dict=True, **mkw)
        last_hidden = outputs.hidden_states[-1]
        h_global, h_contrast, h_focus = model.extract_stage_hiddens(inputs.input_ids, last_hidden)
        semantic_probes, film_mods, (g_out, c_out, f_out) = model.adapter(h_global, h_contrast, h_focus)
        pred_logits = model.decode_mask_sam(backbone_out, semantic_probes, film_mods)
        per_item_cf = [{} for _ in range(B)]
        if model.use_counterfactual and model.cf_heads is not None:
            gt_sam = gt_mask_tensor.unsqueeze(0).repeat(B,1,1,1).to(
                device=g_out.device, dtype=g_out.dtype)
            per_item_cf = model.compute_stage_qualities(g_out, c_out, f_out, gt_sam)
        for i, comp in enumerate(sub):
            r_fmt = format_reward_baseline(comp)
            pred_bboxes = extract_bboxes_from_text(comp)
            r_bbox = bbox_reward(pred_bboxes, gt_bboxes) if W["bbox"] > 0 else 0.0
            pred_mask = torch.sigmoid(pred_logits[i].float()).cpu() if pred_logits is not None else None
            r_mask = mask_reward(pred_mask, gt_mask_tensor.cpu()) if pred_mask is not None else 0.0
            think_match = re.search(r"<think>(.*?)</think>", comp, re.DOTALL)
            rt = think_match.group(1).strip() if think_match else ""
            r_reason, _ = reasoning_reward(rt, pred_bboxes, image_size=model.sam_image_size)
            cf_i = per_item_cf[i] if i < len(per_item_cf) else {}
            r_stage = float(np.mean([cf_i.get("global_cf_dice",0.0),
                                     cf_i.get("contrast_cf_dice",0.0),
                                     cf_i.get("focus_cf_dice",0.0)])) if cf_i else 0.0
            cons = consistency_reward(comp, pred_bboxes, image_size=model.sam_image_size)

            r_cf_mask = 0.0
            if use_cf_mask and diff_weight > 0 and pred_mask is not None:
                stripped_mask = counterfactual_mask_dice(
                    model, processor, image, query, comp, backbone_out,
                    gt_mask_tensor, policy_device)
                if stripped_mask is not None:
                    d_full = _dice(pred_mask, gt_mask_tensor.cpu())
                    d_stripped = _dice(stripped_mask, gt_mask_tensor.cpu())
                    r_cf_mask = max(0.0, min(1.0, d_full - d_stripped)) * diff_weight

            if reward_mode == "baseline":  r_base = r_fmt + r_bbox
            elif reward_mode == "no_stage": r_base = 0.05*r_fmt+0.25*r_bbox+0.45*r_mask+0.25*r_reason
            else: r_base = (W["fmt"]*r_fmt+W["bbox"]*r_bbox+W["mask"]*r_mask+
                           W["reasoning"]*r_reason+cf_mask_weight*r_cf_mask+
                           W["stage"]*r_stage+W["consistency"]*cons["r_consistency"])
            gated = False
            if reward_mode != "baseline" and r_fmt < gate_thr:
                r_total = FORMAT_GATE_BONUS; fmt_gated += 1; gated = True
            else: r_total = r_base
            all_rewards.append(r_total)
            all_details.append({"r_format":r_fmt,"r_bbox":r_bbox,"r_mask":r_mask,
                "r_reasoning":r_reason,"r_cf_mask":r_cf_mask,"r_stage":r_stage,
                "r_consistency":cons["r_consistency"],"format_gated":gated})
        del outputs, inputs
    return all_rewards, all_details, fmt_gated


# ==============================================================
# GTPO Conflict-Aware Gradient Mask (GPU-vectorized)
# ==============================================================
def _build_gtpo_mask(input_ids, labels, tokenizer, structural_token_ids):
    B, L = input_ids.shape
    mask = torch.ones(B, L - 1, device=input_ids.device)
    ids_slice = input_ids[:, 1:]
    for tok_id in structural_token_ids:
        mask = torch.where(ids_slice == tok_id, 0.0, mask)
    return mask


# ==============================================================
# Token Entropy Monitor
# ==============================================================
@torch.no_grad()
def _compute_entropy(logits, labels, ignore_index=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T_m1, V = shift_logits.shape
    total_neg_lp = 0.0
    total_valid = 0
    STRIDE = 2
    for b in range(0, B, STRIDE):
        be = min(b + STRIDE, B)
        sl_b = shift_logits[b:be]
        slb_b = shift_labels[b:be]
        valid = (slb_b != ignore_index)
        n_v = valid.sum().item()
        if n_v == 0:
            continue
        bs = be - b
        nll_b = F.cross_entropy(
            sl_b.float().reshape(bs * T_m1, V),
            slb_b.clamp(min=0).reshape(bs * T_m1),
            reduction='none',
        ).reshape(bs, T_m1)
        total_neg_lp += (nll_b * valid.float()).sum().item()
        total_valid += int(n_v)
    if total_valid == 0:
        return 0.0
    return float(total_neg_lp / total_valid)


# ==============================================================
# Policy forward (V4 prompt + GTPO)
# ==============================================================
def policy_forward(model, ref_mllm, processor, image, query, completions,
                   advantages, mllm_device, kl_coeff, sub_batch_size,
                   fmt_consistency_weight=0.02, structural_token_ids=None):
    G = len(completions); total_pl = torch.tensor(0.0, device=mllm_device); n_sub = 0
    total_entropy = 0.0
    marker_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    for start in range(0, G, sub_batch_size):
        end = min(start + sub_batch_size, G)
        sub = completions[start:end]; sub_adv = advantages[start:end]; B = len(sub)
        texts = []
        for comp in sub:
            msgs = _build_training_messages_v4(image, query, comp)
            texts.append(processor.apply_chat_template(msgs, tokenize=False,
                         add_generation_prompt=False))
        inputs = processor(text=texts, images=[image]*B, padding=True,
                           return_tensors="pt").to(mllm_device)
        mkw = {}
        if "image_grid_thw" in inputs: mkw["image_grid_thw"] = inputs["image_grid_thw"]
        labels = inputs.input_ids.clone()
        for b in range(B):
            ids = inputs.input_ids[b]; found = False
            for i in range(len(ids)-len(marker_ids)+1):
                if ids[i:i+len(marker_ids)].tolist() == marker_ids:
                    labels[b,:i+len(marker_ids)] = -100; found = True; break
            if not found: labels[b,:len(ids)//2] = -100

        outputs = model.mllm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                             pixel_values=inputs.get("pixel_values"),
                             return_dict=True, **mkw)
        logits = outputs.logits

        total_entropy += _compute_entropy(logits, labels)

        sl = logits[:,:-1,:].contiguous(); sll = labels[:,1:].contiguous()
        lpa = F.log_softmax(sl, dim=-1)
        tlp = lpa.gather(dim=-1, index=sll.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        tm = (sll != -100).float()

        gtpo_mask = _build_gtpo_mask(inputs.input_ids, labels,
                                     processor.tokenizer, structural_token_ids)

        fmt_bonus = torch.tensor(0.0, device=mllm_device)
        if fmt_consistency_weight > 0:
            structural_mask = (gtpo_mask < 0.5).float()
            fmt_valid = structural_mask * tm
            fmt_n = fmt_valid.sum(dim=-1) + 1e-8
            fmt_lp = (tlp * fmt_valid).sum(dim=-1) / fmt_n
            fmt_bonus = fmt_consistency_weight * fmt_lp.mean()

        adv_t = torch.tensor(sub_adv, device=mllm_device, dtype=torch.float32)
        for b in range(B):
            if adv_t[b] > 0:
                gtpo_mask[b][gtpo_mask[b] < 0.5] = 2.0

        tlp_masked = tlp * gtpo_mask
        pslp = (tlp_masked * tm).sum(dim=-1) / ((tm * (gtpo_mask > 0.1).float()).sum(dim=-1) + 1e-8)

        if ref_mllm is not None and kl_coeff > 0:
            rd = next(ref_mllm.parameters()).device
            with torch.no_grad():
                ro = ref_mllm(input_ids=inputs.input_ids.to(rd),
                              attention_mask=inputs.attention_mask.to(rd),
                              pixel_values=inputs.get("pixel_values").to(rd) if inputs.get("pixel_values") is not None else None,
                              **{k:v.to(rd) if isinstance(v,torch.Tensor) else v for k,v in mkw.items()})
                rsl = ro.logits[:,:-1,:].contiguous()
                rlpa = F.log_softmax(rsl, dim=-1)
                rtlp = rlpa.gather(dim=-1, index=sll.to(rd).clamp(min=0).unsqueeze(-1)).squeeze(-1)
                gtpo_mask_ref = (gtpo_mask > 0.1).float().to(rd)
                rpslp = (rtlp * tm.to(rd) * gtpo_mask_ref).sum(dim=-1) / \
                        ((tm.to(rd) * gtpo_mask_ref).sum(dim=-1) + 1e-8)
            rpslp = rpslp.to(mllm_device)
            kl = 0.5*(pslp - rpslp).pow(2).mean()
        else: kl = torch.tensor(0.0, device=mllm_device)

        total_pl = total_pl - (adv_t * pslp).mean() + kl_coeff * kl - fmt_bonus; n_sub += 1
        del outputs, inputs
        if ref_mllm is not None: del ro

    avg_ent = total_entropy / max(n_sub, 1)
    return total_pl / max(n_sub, 1), avg_ent


# ==============================================================
# Light seg loss on generated completions (keeps adapter/SAM in sync with LoRA)
# ==============================================================
def seg_forward(model, processor, image, query, top_comps, gt_mask_tensor,
                backbone_out, device):
    if not top_comps:
        return torch.tensor(0.0, device=device)
    texts = []
    for comp in top_comps:
        msgs = _build_training_messages_v4(image, query, comp)
        texts.append(processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=False))
    inputs = processor(text=texts, images=[image] * len(top_comps),
                       padding=True, return_tensors="pt").to(device)
    mkw = {}
    if "image_grid_thw" in inputs:
        mkw["image_grid_thw"] = inputs["image_grid_thw"]
    out = model.mllm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                     pixel_values=inputs.get("pixel_values"),
                     output_hidden_states=True, return_dict=True, **mkw)
    hg, hc, hf = model.extract_stage_hiddens(inputs.input_ids, out.hidden_states[-1])
    probes, film, _ = model.adapter(hg, hc, hf)
    pred_logits = model.decode_mask_sam(backbone_out, probes, film)
    gt = gt_mask_tensor.to(pred_logits.device).repeat(len(top_comps), 1, 1, 1)
    seg_loss = model._dice_ce_loss(pred_logits, gt)
    del out, inputs
    return seg_loss


# ==============================================================
# Evaluation 
# ==============================================================
@torch.no_grad()
def evaluate_sa_v4(model, test_dataset, processor, mllm_device, desc="Eval", max_samples=None):
    model.eval(); total_dice = 0.0; count = 0
    indices = list(range(len(test_dataset)))
    if max_samples and max_samples > 0 and max_samples < len(indices):
        import random; rng = random.Random(42); indices = rng.sample(indices, max_samples)
    pbar = tqdm(indices, desc=desc)
    for i in pbar:
        item = test_dataset.data[i]; img = PILImage.open(item["image_path"]).convert("RGB")
        gt = np.array(PILImage.open(item["mask_path"]).convert("L").resize(
            (model.sam_image_size,model.sam_image_size), PILImage.NEAREST)).astype(np.float32)/255.0
        msgs = _build_inference_messages_v4(img, item["query"])
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ins = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(mllm_device)
        gids = model.mllm.generate(**ins, max_new_tokens=512, do_sample=False)
        gen_text = processor.batch_decode(gids[:, ins.input_ids.shape[1]:], skip_special_tokens=False)[0]
        msgs_f = _build_training_messages_v4(img, item["query"], gen_text)
        tf = processor.apply_chat_template(msgs_f, tokenize=False, add_generation_prompt=False)
        ins_f = processor(text=[tf], images=[img], padding=True, return_tensors="pt").to(mllm_device)
        mkw = {}
        if "image_grid_thw" in ins_f: mkw["image_grid_thw"] = ins_f["image_grid_thw"]
        out = model.mllm(input_ids=ins_f.input_ids, attention_mask=ins_f.attention_mask,
                         pixel_values=ins_f.get("pixel_values"),
                         output_hidden_states=True, return_dict=True, **mkw)
        hg, hc, hf = model.extract_stage_hiddens(ins_f.input_ids, out.hidden_states[-1])
        probes, film, _ = model.adapter(hg, hc, hf)
        si = img.resize((model.sam_image_size, model.sam_image_size))
        st = torch.from_numpy(np.array(si).astype(np.float32).transpose(2,0,1)/255.0).unsqueeze(0)
        bo = model.encode_image_sam(st); pl = model.decode_mask_sam(bo, probes, film)
        pred = (torch.sigmoid(pl[0,0].float()).cpu().numpy()>0.5).astype(np.float32)
        inter = (pred*gt).sum(); uv = pred.sum()+gt.sum()
        total_dice += float(2*inter/uv if uv>0 else 1.0); count += 1
        pbar.set_postfix({"dice": f"{total_dice/count:.4f}"})
        del out, ins, ins_f; torch.cuda.empty_cache()
    return total_dice / max(count, 1)


# ==============================================================
# Main
# ==============================================================
def main():
    args = parse_args()
    if args.no_counterfactual: args.use_counterfactual = False
    if args.batch_gen <= 0: args.batch_gen = args.group_size
    pd = torch.device(f"cuda:{args.policy_device}"); sd = torch.device(f"cuda:{args.sam_device}")
    W = REWARD_WEIGHTS

    args.use_cf_mask = not args.no_cf_mask
    W["cf_mask"] = args.cf_mask_weight if args.use_cf_mask else 0.0



    global DIFFICULTY_WEIGHT
    if args.cf_diff_weights:
        for kv in args.cf_diff_weights.split(","):
            k, v = kv.split(":")
            DIFFICULTY_WEIGHT[k.strip()] = float(v.strip())

    print("=" * 65)
    print("  Stage-Aware Stage 2: GRPO (Per-Stage Seg Tokens)")
    print("=" * 65)
    print(f"  Stage1: {args.stage1_ckpt}  |  LR: {args.lr}  |  KL: {args.kl_coeff}")
    print(f"  Reward: {W['fmt']}*fmt+{W['bbox']}*bbox+{W['mask']}*mask+{W['cf_mask']}*cf_mask+{W['stage']}*stage+{W['consistency']}*cons")
    print(f"  CF-Mask: {'ON' if args.use_cf_mask else 'OFF'} | diff_weights={DIFFICULTY_WEIGHT}")
    gate_info = "DISABLED" if args.no_gate else f"THR={FORMAT_GATE_INIT}→{FORMAT_GATE_FINAL}"
    print(f"  Gate: {gate_info}")
    print(f"  GTPO: conflict mask ON | Entropy: tracked per batch")
    ent_info = f"HALT@{args.entropy_halt_threshold}" if args.entropy_halt else "OFF"
    fmt_info = f"{args.format_consistency_weight}" if args.format_consistency_weight > 0 else "OFF"
    print(f"  EntropyHalt: {ent_info} | FormatConsistency: {fmt_info}")
    print(f"  Speed: batch_gen={args.batch_gen}/{args.group_size} | max_new={args.max_new_tokens}")
    print(f"  Device: policy=cuda:{args.policy_device}, SAM=cuda:{args.sam_device}")
    print("=" * 65)

    global GATE_ENABLED
    if args.no_gate:
        GATE_ENABLED = False
        print("[Gate] DISABLED — GTPO conflict mask active")
    else:
        GATE_ENABLED = True
        print("[Gate] ENABLED (threshold 0.30→0.50 over 2400 steps)")

    # ── Load V4 model ──
    from model.core_seg_stage_aware_v4 import CORESegStageAwareV4
    pm = CORESegStageAwareV4(mllm_path=args.mllm_path, sam_path=args.sam_path,
        sam_ckpt=args.sam_ckpt, sam_config=args.sam_config, sam_dtype=args.sam_dtype,
        adapter_tokens_per_stage=args.adapter_tokens_per_stage,
        adapter_output_tokens=args.adapter_output_tokens,
        adapter_resmlp_layers=args.adapter_resmlp_layers,
        use_film=not args.no_film,
        use_dialogue=not args.no_dialogue,
        use_counterfactual=args.use_counterfactual)
    print("[Model] CORESegStageAwareV4")

    # ── Build GTPO token set (programmatic, includes <seg_*> tokens) ──
    v4_gtpo_ids = _build_v4_gtpo_set(pm.processor)
    print(f"[GTPO] V4 structural token IDs: {len(v4_gtpo_ids)} tokens")

    cd = args.stage1_ckpt
    if os.path.exists(os.path.join(cd, "adapter_config.json")):
        pm.mllm = pm.mllm.cpu(); torch.cuda.empty_cache()
        pm.mllm.load_adapter(cd, adapter_name="default"); print("[Load] LoRA")
    _emb = os.path.join(cd, "embed_tokens.pth")
    if os.path.exists(_emb):
        pm.mllm.get_input_embeddings().weight.data.copy_(torch.load(_emb, map_location="cpu"))
        _lm = os.path.join(cd, "lm_head.pth")
        if os.path.exists(_lm):
            pm.mllm.get_output_embeddings().weight.data.copy_(torch.load(_lm, map_location="cpu"))
        print("[Load] embedding")
    else:
        print("[Load] ⚠️")
    for fname, attr, desc in [("adapter.pth","adapter","Adapter"),
                               ("sam_mask_decoder.pth","sam_mask_decoder","SAM dec"),
                               ("cf_heads.pth","cf_heads","CF heads")]:
        fp = os.path.join(cd, fname)
        if os.path.exists(fp):
            if fname == "cf_heads.pth" and not args.use_counterfactual: continue
            st = torch.load(fp, map_location="cpu")
            if args.sam_dtype != "fp32" and fname != "cf_heads.pth":
                dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.sam_dtype]
                st = {k: v.to(dt) for k, v in st.items()}
            getattr(pm, attr).load_state_dict(st); print(f"[Load] {desc}")

    embed = pm.mllm.get_input_embeddings(); embed.weight.requires_grad = True
    pm.mllm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.mllm = pm.mllm.to(pd)
    pm.sam_image_encoder = pm.sam_image_encoder.to(sd)
    pm.sam_mask_decoder = pm.sam_mask_decoder.to(sd)
    pm.sam_prompt_encoder = pm.sam_prompt_encoder.to(sd)
    pm.adapter = pm.adapter.to(sd)
    if pm.cf_heads is not None: pm.cf_heads = pm.cf_heads.to(sd)

    from data.dataset import CoreSegDataset; proc = pm.processor
    train_ds = CoreSegDataset(data_path=args.train_data, processor=proc,
                              sam_image_size=pm.sam_image_size)
    test_ds = CoreSegDataset(data_path=args.test_data, processor=proc,
                             sam_image_size=pm.sam_image_size)
    print(f"[Data] Train:{len(train_ds.data)} Test:{len(test_ds.data)}")
    if args.train_samples > 0 and args.train_samples < len(train_ds.data):
        import random as _r; _r.seed(42)
        train_ds.data = _r.sample(train_ds.data, args.train_samples)

    tps = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tps, lr=args.lr, weight_decay=0.01)
    spe = len(train_ds) // args.grad_accum_steps
    if len(train_ds) % args.grad_accum_steps: spe += 1
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spe * args.epochs)

    ref_mllm = None
    ref_device = torch.device(f"cuda:{args.ref_device if args.ref_device >= 0 else args.sam_device}")
    if args.kl_coeff > 0:
        from transformers import Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
        rb = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.mllm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        rb.resize_token_embeddings(len(proc.tokenizer))
        ref_mllm = PeftModel.from_pretrained(rb, cd, adapter_name="default").merge_and_unload()
        ref_mllm = ref_mllm.to(ref_device).eval()
        for p in ref_mllm.parameters(): p.requires_grad = False
        print(f"[Setup] Ref MLLM → {ref_device}")

    start_epoch = 0; best_dice = 0.0
    if args.resume:
        rdir = args.resume; rmd = os.path.join(rdir, "best_model")
        if os.path.exists(os.path.join(rmd, "adapter_config.json")):
            pm.mllm = pm.mllm.cpu(); torch.cuda.empty_cache()
            pm.mllm.load_adapter(rmd, adapter_name="default")
        _emb = os.path.join(rmd, "embed_tokens.pth")
        if os.path.exists(_emb):
            pm.mllm.get_input_embeddings().weight.data.copy_(torch.load(_emb, map_location="cpu"))
            _lm = os.path.join(rmd, "lm_head.pth")
            if os.path.exists(_lm):
                pm.mllm.get_output_embeddings().weight.data.copy_(torch.load(_lm, map_location="cpu"))
            print("[Resume] embedding load")
        for fname, attr in [("adapter.pth","adapter"),("sam_mask_decoder.pth","sam_mask_decoder"),
                            ("cf_heads.pth","cf_heads")]:
            fp = os.path.join(rmd, fname)
            if os.path.exists(fp):
                if fname == "cf_heads.pth" and pm.cf_heads is None: continue
                st = torch.load(fp, map_location="cpu")
                if args.sam_dtype != "fp32" and fname != "cf_heads.pth":
                    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.sam_dtype]
                    st = {k: v.to(dt) for k, v in st.items()}
                getattr(pm, attr).load_state_dict(st)
        pm.mllm = pm.mllm.to(pd); pm.sam_image_encoder = pm.sam_image_encoder.to(sd)
        pm.sam_mask_decoder = pm.sam_mask_decoder.to(sd); pm.sam_prompt_encoder = pm.sam_prompt_encoder.to(sd)
        pm.adapter = pm.adapter.to(sd)
        if pm.cf_heads is not None: pm.cf_heads = pm.cf_heads.to(sd)
        op = os.path.join(rdir, "optimizer.pt")
        if os.path.exists(op):
            try:
                os_ = torch.load(op, map_location='cpu')
                opt.load_state_dict(os_["optimizer"]); sch.load_state_dict(os_["scheduler"])
                start_epoch = os_.get("epoch",0); best_dice = os_.get("best_dice",0.0)
                print(f"[Resume] epoch={start_epoch} best={best_dice:.4f}")
            except Exception as e: print(f"[Resume] opt corrupt ({e})")
    elif not args.skip_s1_eval:
        s1d = evaluate_sa_v4(pm, test_ds, proc, pd, desc="S1 baseline", max_samples=args.test_samples)
        print(f"[S1 Eval] Dice: {s1d:.4f}"); best_dice = s1d

    os.makedirs(args.output_dir, exist_ok=True)

    def _save(m, sd, ep, stp, dice, opt, sch):
        os.makedirs(sd, exist_ok=True); m.mllm.save_pretrained(sd)
        _in = m.mllm.get_input_embeddings()
        torch.save(_in.weight.detach().cpu(), os.path.join(sd, "embed_tokens.pth"))
        _out = m.mllm.get_output_embeddings()
        if _out is not None and _out.weight is not _in.weight:
            torch.save(_out.weight.detach().cpu(), os.path.join(sd, "lm_head.pth"))
        torch.save(m.adapter.state_dict(), os.path.join(sd, "adapter.pth"))
        torch.save(m.sam_mask_decoder.state_dict(), os.path.join(sd, "sam_mask_decoder.pth"))
        if m.cf_heads is not None:
            torch.save(m.cf_heads.state_dict(), os.path.join(sd, "cf_heads.pth"))
        torch.save({"optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
                     "epoch": ep, "step": stp, "dice": dice}, os.path.join(sd, "optimizer.pt"))

    global_step = 0
    ENT_WINDOW = 200
    ent_history = deque(maxlen=ENT_WINDOW)
    cf_by_diff = {}
    total_steps = len(train_ds) * (args.epochs - start_epoch)
    print(f"[KL] Anneal: {args.kl_coeff} → {args.kl_anneal_to} over {total_steps} steps"
          if args.kl_anneal_to >= 0 else f"[KL] Fixed: {args.kl_coeff}")
    for epoch in range(start_epoch, args.epochs):
        pm.train(); indices = list(range(len(train_ds))); np.random.shuffle(indices)
        tp = tr = tfmt = tbbox = tmask = treas = tcfmask = tstage = tcons = tgated = tent = 0.0
        n = 0; opt.zero_grad()
        pbar = tqdm(indices, desc=f"V4 Stage2 E{epoch+1}/{args.epochs}")
        for step, idx in enumerate(pbar):
            item = train_ds.data[idx]; img = PILImage.open(item["image_path"]).convert("RGB")
            query = item["query"]; gtb = item["bboxes_512"]
            gmp = PILImage.open(item["mask_path"]).convert("L")
            gmr = gmp.resize((pm.sam_image_size, pm.sam_image_size), PILImage.NEAREST)
            gmt = torch.from_numpy(np.array(gmr).astype(np.float32)/255.0).unsqueeze(0)

            _diff = (item.get("meta") or {}).get("difficulty", "medium")
            diff_weight = get_diff_weight(_diff, global_step, total_steps)

            pm.eval()
            comps = generate_rollouts_batched(pm, proc, img, query, args.group_size,
                                               args.temperature, args.max_new_tokens,
                                               pd, args.batch_gen)
            pm.train()

            si = img.resize((pm.sam_image_size, pm.sam_image_size))
            st = torch.from_numpy(np.array(si).astype(np.float32).transpose(2,0,1)/255.0).unsqueeze(0)
            bo = pm.encode_image_sam(st)

            rewards, details, fg = reward_forward(
                pm, proc, img, query, comps, gtb, gmt, pd,
                args.reward_mode, args.sub_batch_size, global_step, bo,
                use_cf_mask=args.use_cf_mask,
                cf_mask_weight=args.cf_mask_weight,
                diff_weight=diff_weight)

            if fg >= args.group_size: advantages = [0.0] * len(rewards)
            else:
                rm = np.mean(rewards); rs = np.std(rewards) + 1e-8
                advantages = [(r-rm)/rs for r in rewards] if np.std(rewards) >= 1e-6 \
                    else [r-0.5 for r in rewards]

            current_kl = get_kl_coeff(global_step, total_steps, args.kl_coeff, args.kl_anneal_to)

            pl, ent = policy_forward(pm, ref_mllm, proc, img, query, comps,
                                advantages, pd, current_kl, args.sub_batch_size,
                                args.format_consistency_weight, v4_gtpo_ids)

            seg_loss = torch.tensor(0.0, device=pd)
            if args.lambda_seg > 0:
                order = np.argsort(rewards)[::-1]
                top_k = max(1, int(len(comps) * args.seg_top_frac))
                top_comps = [comps[i] for i in order[:top_k]]
                seg_loss = seg_forward(pm, proc, img, query, top_comps, gmt, bo, pd)

            loss = pl / args.grad_accum_steps + args.lambda_seg * seg_loss.to(pd)
            loss.backward()
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(tps, max_norm=1.0)
                opt.step(); sch.step(); opt.zero_grad()

            tp += pl.item(); tr += np.mean(rewards); n += 1
            tfmt += np.mean([d["r_format"] for d in details])
            tbbox += np.mean([d["r_bbox"] for d in details])
            tmask += np.mean([d["r_mask"] for d in details])
            treas += np.mean([d["r_reasoning"] for d in details])
            tcfmask += np.mean([d.get("r_cf_mask", 0.0) for d in details])
            cf_by_diff.setdefault(_diff, []).append(np.mean([d.get("r_cf_mask", 0.0) for d in details]))

            if (step + 1) % CF_LOG_INTERVAL == 0 and cf_by_diff:
                _s = "  ".join(f"{k}={np.mean(v):.4f}(n={len(v)})" for k, v in sorted(cf_by_diff.items()))
                tqdm.write(f"  [cf_by_diff] step {global_step}: {_s}")
            tstage += np.mean([d.get("r_stage",0) for d in details])
            tcons += np.mean([d.get("r_consistency",0) for d in details])
            tgated += fg; tent += ent

            ent_history.append(ent)
            if args.entropy_halt and len(ent_history) >= ENT_WINDOW // 2 and global_step > 200:
                w_ent = sum(ent_history) / len(ent_history)
                w_fmt = tfmt / max(n, 1)
                if w_ent > args.entropy_halt_threshold and w_fmt < 0.85:
                    tqdm.write(
                        f"\n{'='*65}\n"
                        f"  [ENTROPY HALT] step {global_step}: window_ent={w_ent:.4f} > {args.entropy_halt_threshold}\n"
                        f"  AND fmt={w_fmt:.3f} < 0.85 — model is drifting.\n"
                        f"  policy={tp/n:.4f} R={tr/n:.3f} mask={tmask/n:.3f}\n"
                        f"  Recovery: increase --kl_coeff, lower --lr, or reload last healthy ckpt\n"
                        f"{'='*65}"
                    )
                    _save(pm, os.path.join(args.output_dir, "emergency_halt"),
                          epoch, global_step, tp/max(n,1), opt, sch)
                    print(f"  [Emergency ckpt] saved → {args.output_dir}/emergency_halt")
                    import sys; sys.exit(1)

            if (step + 1) % args.log_interval == 0:
                pbar.set_postfix({"policy": f"{tp/n:.4f}", "R": f"{tr/n:.3f}",
                    "fmt": f"{tfmt/n:.3f}", "bbox": f"{tbbox/n:.3f}",
                    "mask": f"{tmask/n:.3f}", "reas": f"{treas/n:.3f}",
                    "cfmask": f"{tcfmask/n:.3f}",
                    "stage": f"{tstage/n:.3f}", "cons": f"{tcons/n:.3f}",
                    "ent": f"{tent/n:.3f}", "gate%": f"{tgated/(n*args.group_size)*100:.0f}"})
            del comps, rewards, advantages, pl, loss, details, bo, ent
            if (step + 1) % 50 == 0: torch.cuda.empty_cache()

            global_step += 1
            if args.ckpt_interval > 0 and global_step % args.ckpt_interval == 0:
                slot = (global_step // args.ckpt_interval) % 2
                _save(pm, os.path.join(args.output_dir, f"ckpt_step{slot}"),
                      epoch, global_step, tp/max(n,1), opt, sch)
                print(f"\n  [Ckpt] step {global_step}, gate={get_gate_threshold(global_step):.2f}")

        dice = evaluate_sa_v4(pm, test_ds, proc, pd, desc=f"Eval epoch {epoch+1}", max_samples=args.test_samples)
        print(f"\n{'='*65}")
        print(f"  Epoch {epoch+1} | Dice: {dice:.4f} | Best: {best_dice:.4f}")
        print(f"  fmt:{tfmt/n:.3f} bbox:{tbbox/n:.3f} mask:{tmask/n:.3f} reas:{treas/n:.3f} cfmask:{tcfmask/n:.3f} stage:{tstage/n:.3f} cons:{tcons/n:.3f}")
        kl_final = get_kl_coeff(global_step, total_steps, args.kl_coeff, args.kl_anneal_to)
        print(f"  Entropy: {tent/n:.4f} | Gate: {tgated/(n*args.group_size)*100:.1f}% gated | KL: {kl_final:.3f}")
        _cf_summ = {k: round(float(np.mean(v)), 3) for k, v in sorted(cf_by_diff.items())}
        print(f"  cf_mask by difficulty: {_cf_summ}")
        cf_by_diff = {}
        print(f"{'='*65}")
        if dice > best_dice:
            best_dice = dice
            _save(pm, os.path.join(args.output_dir, "best_model"), epoch+1, global_step, best_dice, opt, sch)
            print(f"  [Best] Dice={dice:.4f}")
        _save(pm, os.path.join(args.output_dir, f"epoch_{epoch+1}"), epoch+1, global_step, best_dice, opt, sch)

    print(f"\n  Done. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
