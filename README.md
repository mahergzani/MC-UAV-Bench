# MC-UAV-Bench

Code and annotations for **MC-UAV-Bench**, a multi-camera tool-calling benchmark for agentic UAV visual question answering (IEEE Internet of Things Journal submission).

- 2,900 question–answer samples on five-camera UAV3D scenes, in six operational categories
- Each sample has its required cameras and a reference tool-call trace
- Evaluation protocols: all cameras (A), oracle cameras (B), and free tool choice (C_tool)
- **UAVAgent**: a ReAct-style reference agent with a Qwen2-VL-2B orchestrator

> **Images are not included.** MC-UAV-Bench is an annotation layer over **UAV3D v1.0**. You need your own copy of UAV3D (see [Bring your own images](#1-bring-your-own-images)).

---

## ⚠️ Disclosure: placeholder values in the tool outputs

The perception tools (`src/tools/tool_executor.py`) are **not a trained perception stack**:

- They prompt the base VLM (Qwen2-VL-2B for UAVAgent in the paper) about the real camera images.
- They turn its answers into structured tool outputs.
- Fields that a single VLM prompt cannot supply are filled with **deterministic placeholder values**, seeded by sample ID and camera so that every run produces the same output.

| Tool | Comes from the VLM, looking at the real image | Deterministic placeholder (seeded per sample) |
|---|---|---|
| `get_camera_view` | (checks that the image exists) | – |
| `detect_objects` | whether each object category is present, and its count | `bbox` pixel coordinates, `position`, `distance_m`; `confidence` is fixed at 0.75 |
| `estimate_depth` | near / medium / far judgement | metric depth is a representative value for that bin, plus seeded jitter |
| `fuse_multi_camera` | object count and hazard judgement across views | `coverage_area_sqm` = 80 × number of cameras |
| `track_motion` | approaching / stationary / receding judgement | `direction_deg`, `predicted_position` |
| `assess_safety` | numeric clearance and open-area estimates; `is_safe` compares them against the task criteria | – |
| `plan_path` | – | the whole path (waypoints, `path_clear`) is synthesized from the seeded RNG |

Also, `execution_time_ms` values in tool results are seeded placeholders, not measured runtimes.

So UAVAgent's results measure a 2B orchestrator driving a VLM-backed tool pipeline that contains placeholder geometry. They do not measure a real detector, depth estimator or planner. As the paper reports, Qwen2-VL-2B and Qwen2-VL-7B did not produce scene-dependent clearance estimates through `assess_safety`.

Under C_tool, models call these same tools through `src/models/react_tool_loop.py`, so the placeholder fields also appear in the tool results they receive. Protocols A and B call no tools.

---

## 1. Bring your own images

1. Get **UAV3D v1.0** from its authors, following their terms of use.
2. Note the directory that *contains* `sweeps/`. The benchmark's image paths are relative to it:
   ```
   <UAV3D_ROOT>/sweeps/town07_row4/CAMERA_FRONT_id_0/1701410531716272896.png
   ```
   Each sample references 5 images, one per camera (`CAM_FRONT`, `CAM_BACK`, `CAM_LEFT`, `CAM_RIGHT`, `CAM_BOTTOM`), drawn from `CAMERA_{FRONT,BACK,LEFT,RIGHT,BOTTOM}_id_*` folders.
3. Point the code at it with **either**:
   ```bash
   export UAV3D_ROOT=/path/to/UAV3D/v1.0
   # or, per command:
   python scripts/step4_run_evaluation.py --image-root /path/to/UAV3D/v1.0 ...
   ```

All image paths are resolved and checked **before** a model is loaded. If any requested camera image is missing, the run stops with `FileNotFoundError`. It never silently falls back to text-only input.

## 2. Installation

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt
```

The paper's open-model results were produced on one NVIDIA A10G (24 GB) with CUDA 13.0. Qwen2-VL is loaded in 4-bit through `bitsandbytes` by default.

## 3. Dataset

```
data/
├── mc_uav_bench_v1.json              # full pool, 2,900 samples
└── processed/
    ├── train_2024.json               # 2,024 samples
    ├── val_425.json                  # 425 samples
    ├── test_451.json                 # 451 samples (every result in the paper)
    └── test_451_safety_subset.json   # 290 safety-critical test samples (FNR / CAR)
```

Every file has the shape `{"split": ..., "count": N, "samples": [...]}`. One sample looks like this:

| Field | Meaning |
|---|---|
| `id` | `MC-UAV_00000` … `MC-UAV_02899` |
| `category` / `sub_category` | one of six categories (below), plus a finer scenario label |
| `difficulty` | `easy` / `medium` / `hard` |
| `question_type` | `open` or `closed` |
| `question`, `answer`, `reasoning` | the QA pair and its reference rationale |
| `required_cameras` | cameras whose evidence the answer needs (≥ 2; 2.56 on average); used by Protocol B |
| `tool_calls` | reference tool-call trace: a list of `{tool, params, purpose, expected_output}` (4.49 calls on average) |
| `image_paths` | `{camera: path relative to UAV3D_ROOT}` for all five cameras |
| `scene_token` | the UAV3D frame the question is about |
| `safety_critical` | `true` iff `category` is a safety-critical category (below) |
| `spatial_measurements`, `requires_3d_reasoning`, `requires_temporal_reasoning` | annotation metadata |
| `quality_metrics`, `verified` | automatic quality scores; `verified` = overall score ≥ 0.65 (2,301 / 2,900) |
| `generation_model`, `generation_timestamp`, `estimated_difficulty_score`, `camera_complexity_score`, `tool_complexity_score` | generation metadata |

**Categories.** The six categories are Landing Area Detection, Entity Detection, Cross-View Reasoning, Obstacle Avoidance, Route Optimization and Weather-Edge Cases. Four of them are **safety-critical**: Landing Area Detection, Entity Detection, Obstacle Avoidance and Weather-Edge Cases. Safety-critical samples make up 64.6% of the pool and 290/451 (64.3%) of the test split.

**Tools in the reference traces.** The traces use the tools listed in Table II of the paper: `get_camera_view`, `detect_objects`, `estimate_depth`, `fuse_multi_camera`, `track_motion`, `assess_safety` and `plan_path`. Their schemas are in `src/tools/tool_registry.py`.

**How the splits were made.**
- The 451 test IDs are the fixed split used for every experiment in the paper.
- The remaining 2,449 samples were split into train (2,024) and val (425), grouped by `scene_token` with a fixed seed. Train and val therefore share no scene.
- 24 train samples come from scenes that also appear in test (different questions, same frame). Val shares no scene with test.
- The train and val splits are provided for future work. No result in the paper uses them.

## 4. Running the evaluation (open-source Qwen models)

Run everything from the repository root with `UAV3D_ROOT` set, or pass `--image-root`. `--test-set` defaults to `data/processed/test_451.json`. Results are written to `outputs/results/`.

**Protocols A and B: end-to-end with all cameras / with oracle cameras**
```bash
for M in qwen2-vl-2b qwen2-vl-7b; do
  python scripts/step4_run_evaluation.py --model $M --protocols A,B \
      --output outputs/results/${M}_protAB.json
done
```

**Protocol C_tool: free tool choice**
```bash
python scripts/step4_run_evaluation.py --model qwen2-vl-2b --protocols C_TOOL \
    --output outputs/results/qwen2-vl-2b_ctool.json
```

**UAVAgent: the fixed tool pipeline with a Qwen2-VL-2B orchestrator**
```bash
python scripts/step4_run_evaluation.py --model uavagent:qwen2-vl-2b --protocols C \
    --output outputs/results/uavagent_qwen2-vl-2b.json
```

Useful flags:
- `--limit N` runs a quick check on the first N samples.
- `--device cuda|cpu` picks the device.

**Scoring note.** `step4_run_evaluation.py` reports the accuracy from `src/evaluation/metrics.py`. The paper's tables used a stricter final scoring rule:
- a response that states neither SAFE nor UNSAFE is never counted correct;
- under C_tool, samples that exhaust the iteration budget or end in an error are counted incorrect.

Because of that rule, some numbers from this script will be higher than the paper's. For example, Qwen2-VL-2B scores 59.20 under Protocol A here (paper: 56.76) and 43.46 under C_tool (paper: 24.83).

## Repository layout

```
src/        data loading (incl. image_root.py), metrics/protocols, model wrappers, UAVAgent, tool registry/executor
scripts/    step4_run_evaluation.py (evaluation entry point)
data/       annotations and splits (no images)
```

## License

The code and annotations are released under the MIT License (see `LICENSE`). The UAV3D images are **not** part of this release; they remain under UAV3D's own terms.

## Citation

```bibtex
@article{mcuavbench2026,
  title   = {MC-UAV-Bench: A Multi-Camera Tool-Calling Benchmark for Agentic UAV Visual Question Answering},
  author  = {Guizani, Maher and Agarwal, Ishita and Kummari, Naveen Kumar and Islam, Mohammad A.},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Under review}
}
```
