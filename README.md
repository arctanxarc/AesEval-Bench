# Can Vision-Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective

This repository contains the evaluation code for DesignBench, a benchmark for studying whether vision-language models can assess graphic design aesthetics.

## Overview

DesignBench evaluates graphic design aesthetics across four dimensions and twelve task categories:

| Dimension | Tasks |
|---|---|
| Layout | `balance`, `layering`, `whitespace`, `alignment` |
| Typography (`font`) | `legibility`, `hierarchy` |
| Graphics | `quality`, `relevance` |
| Color | `harmony`, `contrast`, `appeal`, `psychology` |

The code supports three evaluation modes:

| `eval_mode` | Description | Expected model output |
|---|---|---|
| `text-preview` | Binary judgment of whether an aesthetic issue exists | `yes` / `no` |
| `text-preview-choice` | Select the problematic region from four options | `A` / `B` / `C` / `D` |
| `text-preview-bbox` | Predict the bounding box of the problematic element | `left,top,width,height` normalized to `[0, 1]` |

## Repository Layout

```text
design_bench/
|-- benchmark_data/          # Benchmark data release
|-- main.py                  # Entry point
|-- evaluator.py             # Evaluation pipeline
|-- models/                  # Model adapters
|-- tasks/                   # Dimension/task definitions
|-- requirements.txt         # Python dependencies
|-- README.md
`-- results/                 # Saved evaluation outputs
```

## Dataset Download

1. Download the DesignBench benchmark release from this [link](https://drive.google.com/file/d/1W5ocLYW0U-znD1Aq3C2xg_TLxL80jeiJ/view?usp=sharing).
2. Extract it under the repository root as `benchmark_data/`.
3. If you store the data elsewhere, pass its location through `--benchmark_path`.

Expected layout:

```text
benchmark_data/
`-- {sample_id}-perturbs_new/
    |-- preview.png
    |-- preview_highlight.png
    |-- simplified_meta_info.json
    |-- GT.json
    |-- changes.json
    |-- 0.png, 1.png, ...
    `-- meta_info.json
```

Notes:

- The evaluator scans sample folders whose names contain `_new`.
- `GT.json` stores labels using keys such as `layout-balance` and `color-harmony`.
- You can point the code to any compatible data directory with `--benchmark_path /path/to/benchmark_data`.

## Installation

```bash
pip install -r requirements.txt
```

## Supported Models

### OpenAI-compatible API models

These models are called through the OpenAI Chat Completions API interface and support concurrent requests.

| `model_type` | Backend model name |
|---|---|
| `gpt-4o` | `gpt-4o` |
| `gpt-4o-mini` | `gpt-4o-mini` |
| `gpt-4.1` | `gpt-4.1` |
| `gpt-4.1-mini` | `gpt-4.1-mini` |
| `gpt-4.1-nano` | `gpt-4.1-nano` |
| `o3` | `o3` |
| `gpt-5` | `gpt-5` |

### Local Qwen2.5-VL models

These models are loaded locally and use `device_map="auto"` for single-GPU or multi-GPU inference.

| `model_type` | Hugging Face checkpoint |
|---|---|
| `qwen2.5-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `qwen2.5-vl-32b` | `Qwen/Qwen2.5-VL-32B-Instruct` |
| `qwen2.5-vl-72b` | `Qwen/Qwen2.5-VL-72B-Instruct` |

By default, the local Qwen adapter sets `HF_ENDPOINT=https://hf-mirror.com`. Override it in your environment if you need a different endpoint.

## Running the Evaluation

### API-based models

Set your credentials through environment variables:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Run evaluation:

```bash
python3 main.py \
    --model_type gpt-4o-mini \
    --eval_mode text-preview \
    --benchmark_path ./benchmark_data \
    --output_dir ./results \
    --max_workers 4
```

Other modes use the same command with a different `--eval_mode`:

```bash
python3 main.py \
    --model_type gpt-4o-mini \
    --eval_mode text-preview-choice \
    --benchmark_path ./benchmark_data \
    --output_dir ./results

python3 main.py \
    --model_type gpt-4o-mini \
    --eval_mode text-preview-bbox \
    --benchmark_path ./benchmark_data \
    --output_dir ./results
```

You can also pass API settings by CLI:

```bash
python3 main.py \
    --model_type gpt-4o \
    --eval_mode text-preview \
    --benchmark_path ./benchmark_data \
    --api_key your_api_key \
    --api_base https://api.openai.com/v1
```

### Local Qwen2.5-VL models

Single GPU:

```bash
python3 main.py \
    --model_type qwen2.5-vl-7b \
    --eval_mode text-preview-choice \
    --benchmark_path ./benchmark_data \
    --output_dir ./results \
    --device_ids 0
```

Multi-GPU automatic sharding:

```bash
python3 main.py \
    --model_type qwen2.5-vl-72b \
    --eval_mode text-preview \
    --benchmark_path ./benchmark_data \
    --output_dir ./results \
    --device_ids 0,1,2,3
```

Use a custom local checkpoint path:

```bash
python3 main.py \
    --model_type qwen2.5-vl-72b \
    --model_path /path/to/Qwen2.5-VL-72B-Instruct \
    --eval_mode text-preview-bbox \
    --benchmark_path ./benchmark_data \
    --output_dir ./results \
    --device_ids 0,1,2,3
```

## Command-Line Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_type` | Model identifier registered in `models/__init__.py` | `gpt-4o-mini` |
| `--model_path` | Local model path for Qwen models | Download from Hugging Face |
| `--api_key` | OpenAI API key for API-based models | `OPENAI_API_KEY` |
| `--api_base` | OpenAI-compatible base URL | `OPENAI_BASE_URL` |
| `--device_ids` | Comma-separated GPU IDs for Qwen models | Auto-detect |
| `--eval_mode` | Evaluation mode | `text-preview` |
| `--benchmark_path` | Path to the benchmark data directory | `./benchmark_data` |
| `--output_dir` | Directory for saved JSON results | `./results` |
| `--max_workers` | Number of concurrent API workers | `4` |

## Output

Each run writes one JSON file to:

```text
results/{model_type}_{eval_mode}.json
```

The output includes:

- Per-task accuracy.
- Per-dimension averages.
- Overall accuracy and sample-level accuracy.
- Prediction details for every evaluated sample.
- API failure counts when applicable.
- Mean IoU for non-`None` ground-truth boxes in `text-preview-bbox` mode.

## Extending the Codebase

### Add a new OpenAI-compatible model

Add the mapping in `models/gpt_model.py`:

```python
MODEL_NAME_MAP = {
    ...,
    "my-model": "my-model-api-name",
}
```

Then register it in `models/__init__.py`:

```python
MODEL_REGISTRY["my-model"] = GPTModel
```

### Add a new Qwen variant

Add the checkpoint mapping in `models/qwen_model.py`:

```python
MODEL_REPO_MAP = {
    ...,
    "qwen2.5-vl-14b": "Qwen/Qwen2.5-VL-14B-Instruct",
}
```
