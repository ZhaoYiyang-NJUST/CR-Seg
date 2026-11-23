# CR-Seg: Counterfactual Reward Learning for Reasoning-Driven Polyp Segmentation

<!-- Add the confirmed author list, affiliations, and paper link here. -->

## 1. Introduction

This repository accompanies **“Counterfactual Reward Learning for Reasoning-Driven Polyp Segmentation.”**

We propose **Counterfactual Reward-based Segmentation (CR-Seg)**, a framework that trains reasoning-driven polyp segmentation using the measured benefit of retaining generated reasoning text.

A high-quality mask alone does not establish whether the reasoning text improves segmentation. CR-Seg compares predictions obtained with the original reasoning text and after removing that text, while retaining structural tags and spatial tokens. The positive Dice difference contributes to training alongside absolute mask quality.

### Highlights

- **Counterfactual mask reward:** directly rewards positive Dice gains from retaining generated reasoning text.
- **Stage-aware spatial readouts:** separate tokens represent global, contrast, and focus reasoning.
- **Cross-stage conditioning:** a dialogue adapter produces sparse prompts and FiLM coefficients for segmentation.
- **Results on the constructed benchmark:** **87.10% overall mean Dice** and **63.22% Hard+VH mean Dice**.

### Contents

- [1. Introduction](#1-introduction)
- [2. Method Overview](#2-method-overview)
- [3. Dataset](#3-dataset)
- [4. Experimental Results](#4-experimental-results)
- [5. Training and Evaluation](#5-training-and-evaluation)
- [6. Citation](#6-citation)
- [7. FAQ](#7-faq)
- [8. Acknowledgments and License](#8-acknowledgments-and-license)

## 2. Method Overview

### 2.1. Framework

CR-Seg uses **Qwen2.5-VL-3B-Instruct** as the multimodal language model and **MedSAM2** as the segmentation backbone.

Given an endoscopic image and a task query, the model generates three reasoning stages:

1. **Global:** the scene and approximate polyp location.
2. **Contrast:** differences between the polyp, surrounding tissue, and confusable structures.
3. **Focus:** boundary and fine-grained morphology.

A dedicated spatial token follows each stage. The cross-stage dialogue adapter refines their hidden states and generates sparse prompts for the mask decoder and FiLM coefficients for visual feature modulation. Each refined stage representation also receives auxiliary localization supervision.

<!--
Insert the framework figure after exporting it to a GitHub-displayable
PNG or SVG and confirming its repository-relative path.

Suggested placement:
<p align="center">
  <img src="CONFIRMED_RELATIVE_PATH" width="100%" alt="Overview of CR-Seg">
</p>
-->
<p align="center">
  <img src="figs/method.pdf" width="100%" alt="Overview of CR-Seg">
</p>

### 2.2. Counterfactual Training

For the same image, query, and checkpoint, CR-Seg evaluates:

- **Full:** the original generated response.
- **Stripped:** the response with only the three stage-wise reasoning texts removed.

Both conditions retain structural tags, spatial tokens, and the bounding-box answer. Contextual representations and masks are recomputed for each condition.

The signed difference is:

$$
\Delta_{\mathrm{CF}}
=
\mathrm{Dice}(M^{\mathrm{full}}, G)
-
\mathrm{Dice}(M^{\mathrm{stripped}}, G).
$$

The positive part of this difference supplies the counterfactual reward, with a difficulty-dependent curriculum weight.

Training consists of:

- **Stage 1:** joint supervision of response generation, segmentation, and auxiliary stage localization.
- **Stage 2:** reinforcement learning with the composite reward, together with segmentation supervision on selected sampled responses.

At inference, CR-Seg generates one response using greedy decoding and predicts the mask from that response. Ground-truth masks and paired text-removal evaluation are not required for ordinary inference. The generated bounding box is a localization output, not a prompt for CR-Seg's mask decoder.

### 2.3. Qualitative Results

Qualitative comparisons and generated-description examples are reported in the manuscript.

<!--
Insert the qualitative comparison figure here after confirming its path.
Use CR-Seg consistently in the image labels and identify adapted
box-to-mask baselines in the caption.
-->

## 3. Dataset

### 3.1. Benchmark Construction

The benchmark is assembled from twelve source collections:

- **23,045** initial image–mask pairs.
- **22,684** pairs after pair-level pixel-hash deduplication.
- **13,518** pairs retained with out-of-fold (OOF) Dice below 95%.
- **1,512** samples excluded during CoT annotation.
- **12,006** samples in the final benchmark.

Original images and their augmented variants are kept in the same split. Frames from the same video sequence are also assigned to the same split.

### 3.2. Source Composition and Splits

The table reports the final retained samples, not the original sizes of the source datasets.

| Source collection | Train | Validation | Test |
|---|---:|---:|---:|
| NeoPolyp-Small | 550 | 65 | 69 |
| CVC-300 | 21 | 0 | 2 |
| CVC-ClinicDB | 92 | 10 | 17 |
| CVC-ColonDB | 80 | 8 | 6 |
| EDD2020 | 48 | 6 | 7 |
| EndoScene | 115 | 12 | 17 |
| EndoScene (pre-augmented) | 3,932 | 492 | 499 |
| ETIS-LaribPolypDB | 27 | 4 | 4 |
| Kvasir-Capsule-SEG | 9 | 2 | 1 |
| Kvasir-SEG | 362 | 44 | 49 |
| PolypDB | 1,950 | 237 | 239 |
| SUN-SEG | 2,475 | 279 | 276 |
| **Total** | **9,661** | **1,159** | **1,186** |

EndoScene and its pre-augmented collection are listed separately to make the source composition explicit.

<!--
Add verified source citations or official dataset links.
Document the provenance of the pre-augmented EndoScene collection.
Only add dataset/split download links when they are available and
redistribution permissions have been checked.
-->

### 3.3. Difficulty and Reasoning Annotations

Difficulty is assigned using five-fold grouped cross-validation with an auxiliary U-Net and ResNet-34 encoder. Each sample is evaluated by the auxiliary model trained on the other four folds.

| Difficulty | OOF Dice (%) |
|---|---|
| Very-hard | Below 60 |
| Hard | 60 to below 80 |
| Medium | 80 to below 90 |
| Easy | 90 to below 95 |

These labels describe difficulty according to auxiliary-model performance.

Each sample contains a task query, a three-stage reasoning annotation, a segmentation mask, and a single reference box enclosing the entire annotated foreground.

Qwen3.6-Plus generates annotations from stage-specific instructions and mask-overlaid images. GPT-4o screens them using five criteria: clinical accuracy, completeness, logical coherence, location agreement, and conciseness. An unweighted mean score of at least 4.0 out of 5 passes automated screening.

Failed annotations are regenerated and rescored up to twice. If all three attempts fail, two clinicians with colonoscopy experience provide the annotation. Both clinicians review every retained annotation and revise it when necessary. Samples with unresolved substantive disagreements are excluded.

## 4. Experimental Results

### 4.1. Main Results

Values are manuscript-reported **mean per-image Dice (%)** on **1,186 test images**. **Hard+VH** is the mean over all **145 hard and very-hard images**.

The following table summarizes pipelines trained on the polyp training split.

| Method | Segmentation backbone | Overall | Hard | Very-hard | Hard+VH |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B (SFT) | MedSAM2 | 84.36 | 74.06 | 46.08 | 59.79 |
| VLM-R1-REC-3B (SFT) | MedSAM2 | 84.65 | 73.60 | 47.22 | 60.14 |
| LISA-7B | SAM | 62.32 | 55.34 | 25.99 | 40.36 |
| MedFuse-Seg | MedSAM | 86.12 | 72.00 | 41.20 | 56.28 |
| **CR-Seg** | MedSAM2 | **87.10** | **77.47** | **49.55** | **63.22** |

The two box-supervised SFT baselines use their first parsed predicted box to prompt MedSAM2, without pixel-level segmentation supervision during adaptation. Invalid-box cases receive Dice 0. LISA and MedFuse-Seg use their native segmentation architectures and supervised mask objectives.

These results compare the specified pipelines, which differ in architecture and supervision. The full comparison, including zero-shot cascades and all difficulty subsets, is reported in the manuscript.

### 4.2. Effect of the Counterfactual Reward

The following comparison retains the stage localization reward and changes whether the counterfactual reward is included.

| Stage-2 configuration | Overall Dice (%) | Hard+VH Dice (%) | Hard text gain (pp) | Very-hard text gain (pp) |
|---|---:|---:|---:|---:|
| Without counterfactual reward | 85.58 | 57.61 | 0.39 | 0.41 |
| **With counterfactual reward** | **87.10** | **63.22** | **4.03** | **6.31** |

**Text gain** is full-minus-stripped Dice for the same model checkpoint, expressed in percentage points (pp), without clipping or difficulty weighting.

Adding the reward increases both full-text and stripped-text Dice on the hard and very-hard subsets, with larger increases when reasoning text is retained.

## 5. Training and Evaluation

### 5.1. Reported Setup

| Setting | Value |
|---|---|
| Framework | PyTorch |
| GPU | NVIDIA RTX 4090 |
| MLLM | Qwen2.5-VL-3B-Instruct |
| Segmentation backbone | MedSAM2, Hiera-T512 configuration |
| Stage-1 epochs / learning rate | 50 / 5e-5 |
| Stage-1 image batch size / gradient accumulation | 1 / 9 |
| Stage-2 epochs / learning rate | 2 / 2e-6 |
| Stage-2 sampled responses per image and query | 4 |
| Sampling temperature / top-p | 0.8 / 0.9 |
| Test-time decoding | Greedy |

The segmentation image encoder is frozen in both stages. The MLLM's LoRA parameters, dialogue adapter, and mask decoder are trainable. Auxiliary localization heads are trained in Stage 1 and frozen in Stage 2.

For the full model, the format, bounding-box, mask, counterfactual, stage-localization, and consistency reward weights are **0.05, 0.20, 0.68, 0.12, 0.10, and 0.05**, respectively.

### 5.2. Evaluation Protocol

- Input: an endoscopic image and a task query without target location or appearance information.
- Output: a segmentation mask derived from one greedily generated response.
- Metric: Dice computed separately for each test image and averaged across images.
- Hard+VH: pooled mean over the hard and very-hard images.

<!--
Add verified installation, model-download, data-preparation, training,
inference, and evaluation commands here.

Commands must match the released code and checkpoints.
Do not substitute generic train.py/test.py commands.
-->

## 6. Citation

**Paper title:** Counterfactual Reward Learning for Reasoning-Driven Polyp Segmentation

<!-- Add the verified paper link and BibTeX after bibliographic details are confirmed. -->

## 7. FAQ

**Does CR-Seg require location descriptions or ground-truth boxes at inference?**

No. The query specifies the task without describing the target's location or appearance. The model's generated box is not used to prompt its mask decoder.

**Is counterfactual evaluation required for every inference image?**

No. Ground-truth-based counterfactual reward evaluation is used during training and diagnostic experiments, not ordinary inference.

**Are the difficulty labels clinical categories?**

No. They are defined by the OOF Dice of an auxiliary segmentation model.

**Does this benchmark use the original source-dataset split sizes?**

The counts above describe the constructed benchmark after deduplication, difficulty filtering, and annotation screening. They are not the original dataset sizes.

## 8. Acknowledgments and License

We acknowledge the developers of Qwen2.5-VL, MedSAM2, and the source datasets used in this study.

Source datasets and pretrained models remain subject to their respective licenses and terms of use.

<!--
Add the repository's actual code license and a link to LICENSE.
Do not infer the code license from the licenses of dependencies or datasets.
-->

[Back to top](#cr-seg-counterfactual-reward-learning-for-reasoning-driven-polyp-segmentation)
