# Learning Transferable Temporal Primitives for Video Reasoning via Synthetic Videos

**CVPR 2026**

[\[Paper\]](https://arxiv.org/abs/xxxx.xxxxx) [\[Project Page\]](https://github.com/jiangsongtao/Synthetic-Video)

## Overview

SynRL is a post-training framework that teaches vision-language models (VLMs) **temporal primitives** — the fundamental building blocks of temporal understanding including direction, speed, acceleration, and state tracking — through programmatically generated synthetic videos with guaranteed ground-truth annotations.

**Key Insight:** Abstract temporal primitives learned from simple synthetic scenarios (geometric shapes, grid puzzles) transfer effectively to complex real-world videos involving human actions, camera motion, and scene dynamics.


## Highlights

- **No proprietary model dependency:** We bypass GPT-4V/Gemini for data synthesis. All training data is generated programmatically with provably correct annotations.
- **21x data efficiency:** 7.7K synthetic CoT samples outperform Video-R1's 165K real-world CoT samples.
- **Consistent gains across 15 benchmarks:** +12.6% on RexTime (temporal grounding), +4.6% on TOMATO (complex reasoning), with improvements on MVBench, VideoMME, TemporalBench, and more.

## Method

SynRL consists of three stages:

1. **Programmatic Video Generation** — Synthetic videos covering short-term perceptual primitives (speed, direction, trajectory, acceleration) and long-term cognitive primitives (state tracking, retrodictive inference) are generated via Python code with frame-level metadata.

2. **Chain-of-Thought Augmentation** — A four-stage pipeline (Generation → Verification → Reflection → Polishing) produces high-quality temporal reasoning chains grounded in procedural metadata.

3. **Two-Stage Training** — (i) SFT on 7.7K CoT samples to teach temporal reasoning structure; (ii) GRPO reinforcement learning on 7K synthetic video-QA pairs with verifiable accuracy rewards.

## Results

### Temporal Grounding

| Model | NExTGQA mIoU | RexTime mIoU | Charades mIoU |
|-------|:---:|:---:|:---:|
| Qwen3-VL-4B | 23.5 | 20.9 | 41.9 |
| **Qwen3-VL-4B + SynRL** | **28.1** (+4.6) | **28.9** (+8.0) | **47.0** (+5.1) |

### Complex Reasoning & General Understanding

| Model | TOMATO | Video-TT | MVBench | VideoMME |
|-------|:---:|:---:|:---:|:---:|
| Qwen3-VL-4B | 32.1 | 38.9 | 65.4 | 60.9 |
| **Qwen3-VL-4B + SynRL** | **36.7** (+4.6) | **40.7** (+1.8) | **67.1** (+1.7) | **62.0** (+1.1) |
| Qwen3-VL-8B | 33.2 | 40.6 | 67.2 | 63.4 |
| **Qwen3-VL-8B + SynRL** | **38.1** (+4.9) | **41.5** (+0.9) | **69.1** (+1.9) | **65.2** (+1.8) |

## Training Data

| Category | Count | Description |
|----------|:-----:|-------------|
| Synthetic CoT (SFT) | 6.7K | Temporal reasoning chains with frame-level annotations |
| Real-world QA (SFT) | 1.0K | LLaVA-Video samples (answer-only, no CoT) |
| Synthetic RL | 7.0K | Video-QA pairs with verifiable rewards |

Synthetic videos span **8 major categories** with **18 subcategories**, covering collision counting, direction identification, trajectory recognition, speed perception, grid-based tracking, symbol manipulation, code execution, and container management.

## Getting Started

### Requirements

- 8x NVIDIA H100 GPUs (80GB)
- Python 3.10+
- VeRL framework for GRPO training


## Acknowledgments

This work builds upon [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) and the [VeRL](https://github.com/volcengine/verl) framework.

## License

This project is released under the [Apache 2.0 License](LICENSE).
