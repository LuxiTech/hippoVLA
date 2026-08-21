# HIPPOVLA: Hippocampus-Inspired Episodic Memory for Long-Horizon Vision-Language-Action Models

HIPPOVLA combines a spatially enhanced vision-language backbone with an explicit hippocampus-inspired short-term memory module, giving Vision-Language-Action (VLA) agents the temporal context needed for long-horizon manipulation.

[![Project Page](https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge&logo=githubpages&logoColor=white)](https://wusimo.github.io/hippoVLA/)

---

HIPPOVLA keeps history outside the language prompt. Sparse historical frames are encoded by the visual stack, summarized by a Short-Term Memory Bank (STMB), and fused with current DeepStack visual features through a learned gate. This lets the policy remember previously observed objects, completed subgoals, and evolving scene configurations while keeping the language context fixed.

<p align="center">
  <img src="https://wusimo.github.io/hippoVLA/assets/abstract.png" alt="HIPPOVLA architecture overview" width="95%">
</p>

## 🔥 Key Features

<details open>
<summary><b>Hippocampus-Inspired Episodic Memory</b></summary>

- [x] **Explicit episodic visual memory:** STMB stores sparsely sampled historical visual features instead of appending history frames to the language prompt.
- [x] **Current-frame grounding and episodic retrieval:** current observations remain the primary visual input while task-relevant short-term history is retrieved separately.
- [x] **Content-aware fusion gate:** current visual tokens query the history buffer through cross-attention, and a learned per-token, per-channel gate determines how strongly memory modifies each token.
- [x] **DeepStack and action-token decoding:** memory-conditioned visual features are injected into intermediate language layers, and dedicated placeholder tokens are decoded into continuous action chunks.

</details>

<details open>
<summary><b>Method Components</b></summary>

### Short-Term Memory Bank (STMB)

STMB operates at the visual-feature level. It samples a sparse history buffer, adds temporal embeddings, aggregates visual history through cross-attention, and produces memory-conditioned features with the same shape as the current visual features.

### Content-Aware Fusion Gate

A learned gate fuses the current observation with aggregated history. High gate values preserve the current frame, while low values incorporate more information from retrieved visual history.

</details>

## 📊 Results

### LIBERO 4-in-1 Memory Ablation

| Setting | Batch | Steps | Base Model | Added Module | Object | Spatial | Goal | Long |
|---|---:|---:|---|---|---:|---:|---:|---:|
| Official | - | 50k | starVLA-Qwen3VL-4B | None | 99.6 | - | 99.0 | 98.6 |
| Reproduced | - | 50k | starVLA-Qwen3VL-4B | None | 99.6 | 93.2 | 98.6 | 93.0 |
| + HIPPOVLA memory (interval 5) | 16 | 20k | starVLA-Qwen3VL-4B | Memory len 5 | 99.6 | 94.2 | 98.2 | 95.8 |
| **+ HIPPOVLA memory (interval 10) [ours]** | **16** | **20k** | **starVLA-Qwen3VL-4B** | **Memory len 5** | **99.8** | **94.4** | **96.4** | **96.0** |

Success rate (%) on the four LIBERO suites under a single 4-in-1 co-training run. With a sampling interval of 10, HIPPOVLA improves Long-horizon by **+3.0** and Spatial by **+1.2** over the reproduced no-memory baseline.

<details open>
<summary><b>CALVIN ABC→D Long-Horizon Leaderboard</b></summary>

| Rank | Method | Input | MTLC | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Avg. Len. |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | FLOWER | Static RGB + Gripper RGB | - | 99.4% | 95.8% | 90.7% | 84.9% | 77.8% | 4.53 |
| **2** | **HIPPOVLA (DTT & Memory) [ours]** | **-** | **-** | **98.9%** | **95.3%** | **89.3%** | **82.9%** | **75.9%** | **4.42** |
| 3 | UniVLA | Static RGB + Gripper RGB | - | 98.9% | 94.8% | 89.0% | 82.8% | 75.1% | 4.41 |
| 4 | HIPPOVLA (MLP & Memory) | - | - | 99.1% | 94.4% | 88.4% | 81.6% | 74.1% | 4.38 |
| 5 | HIPPOVLA (MLP & Circular Memory) | - | - | 97.1% | 88.7% | 80.7% | 72.1% | 63.5% | 4.376 |
| 6 | SeeR-Large | Static RGB + Gripper RGB + Proprio | - | 96.3% | 91.6% | 86.1% | 80.3% | 74.0% | 4.28 |
| 7 | GR-MG | Static RGB + Gripper RGB + Proprio | - | 96.8% | 89.3% | 81.5% | 72.7% | 64.4% | 4.04 |
| 8 | MoDE | Static RGB + Gripper RGB | - | 96.2% | 88.9% | 81.1% | 71.8% | 63.5% | 4.01 |
| 9 | RynnBrain (MLP & No Memory) | - | - | 88.7% | 79.9% | 72.7% | 67.3% | 61.0% | 3.70 |
| 10 | GHIL-Glue | Static RGB | - | 95.2% | 88.5% | 73.2% | 62.5% | 49.8% | 3.69 |
| 11 | RoboUniView | Static RGB-D + Gripper RGB-D + Cam params | - | 94.2% | 84.2% | 73.4% | 62.2% | 50.7% | 3.64 |
| 12 | Diffusion Transformer Policy | Static RGB | - | 94.5% | 82.5% | 72.8% | 61.3% | 50.0% | 3.61 |
| 13 | CLOVER | Static RGB-D | - | 96.0% | 83.5% | 70.8% | 57.5% | 45.4% | 3.53 |
| 14 | 3D Diffuser Actor | Static RGB-D + Gripper RGB-D + Proprio + Cam params | - | 92.2% | 78.7% | 63.9% | 51.2% | 41.2% | 3.27 |
| 15 | GR-1 | Static RGB + Gripper RGB + Proprio | - | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06 |
| 16 | DeeR | Static RGB + Gripper RGB | - | 86.2% | 70.1% | 51.8% | 41.5% | 30.4% | 2.82 |
| 17 | SuSIE | Static RGB | - | 87.0% | 69.0% | 49.0% | 38.0% | 26.0% | 2.69 |
| 18 | RoboFlamingo | Static RGB + Gripper RGB | - | 82.4% | 61.9% | 46.6% | 33.1% | 23.5% | 2.47 |
| 19 | SPIL | Static RGB + Gripper RGB | - | 74.2% | 46.3% | 27.6% | 14.7% | 8.0% | 1.71 |
| 20 | HULC | Static RGB + Gripper RGB | - | 41.8% | 16.5% | 5.7% | 1.9% | 1.1% | 0.67 |
| 21 | Baseline | Static + Gripper RGB | 38.0% | 30.4% | 1.3% | 0.17% | 0.0% | 0.0% | 0.31 |
| 22 | Baseline | Static + Tactile | 43.7% | 17.3% | 0.8% | 0.08% | 0.0% | 0.0% | 0.26 |
| 23 | Baseline | Static RGB-D + Gripper RGB-D | 30.8% | 21.1% | 1.3% | 0.0% | 0.0% | 0.0% | 0.22 |
| 24 | Baseline | Static RGB | 38.6% | 20.2% | 0.2% | 0.0% | 0.0% | 0.0% | 0.20 |

HIPPOVLA reaches rank 2 with an average length of **4.42**, improving by **+0.72** over the no-memory ablation (3.70). Step 5 success rises from **61.0%** to **75.9%**.

</details>

## 🎬 Rollout Videos

| Benchmark | Task | Video |
|---|---|---|
| LIBERO-Spatial | Pick the bowl between the plate and ramekin and place it on the plate | [Watch rollout](https://wusimo.github.io/hippoVLA/assets/spatial_demo.mp4) |
| LIBERO-Object | Pick up the alphabet soup and place it in the basket | [Watch rollout](https://wusimo.github.io/hippoVLA/assets/object_demo.mp4) |
| LIBERO-Goal | Open the middle drawer of the cabinet | [Watch rollout](https://wusimo.github.io/hippoVLA/assets/goal_demo.mp4) |
| LIBERO-10 (Long) | Pick up the book and place it in the back compartment of the caddy | [Watch rollout](https://wusimo.github.io/hippoVLA/assets/long_demo.mp4) |

## 🤖 Real-World Experiments

HIPPOVLA is evaluated on a repeated pick-and-place task: pick up the blue cube, place it on the target, and repeat the action sequence. The comparison shows how episodic memory helps the policy track completed subgoals and continue a long-horizon manipulation sequence.

| Setting | Result | Video |
|---|---|---|
| **With memory** | Repeats the blue-cube pick-and-place three times | [Watch experiment](https://wusimo.github.io/hippoVLA/assets/PickXtimes_memory.mp4) |
| **Without memory** | Repeats the blue-cube pick-and-place only two times | [Watch experiment](https://wusimo.github.io/hippoVLA/assets/PickXtimes_no_memory.mp4) |

## 📖 FAQ

<details close>
<summary><b>Why keep history outside the language prompt?</b></summary>

The memory pathway handles sparse visual history at the feature level, preserving a fixed language context while still giving the policy access to past observations and completed subgoals.

</details>

<details close>
<summary><b>How is memory fused with the current observation?</b></summary>

Current visual features query the temporal history buffer through cross-attention. A learned per-token, per-channel gate then balances current-frame information against the retrieved memory.

</details>

<details close>
<summary><b>Where does episodic memory help most?</b></summary>

The reported gains are strongest on long-horizon tasks. On LIBERO-10 (Long), the interval-10 memory configuration improves the reproduced baseline from 93.0 to 96.0. On CALVIN ABC→D, the DTT-and-memory variant improves average sequence length from 3.70 to 4.42 over the no-memory ablation.

</details>


## 🔗 Project Page

For the original interactive presentation and embedded videos, visit [wusimo.github.io/hippoVLA](https://wusimo.github.io/hippoVLA/).
