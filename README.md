# MedGemma ECG Batch Inference on Snowflake

Distributed multi-modal inference using Google's MedGemma 4B vision model to classify ECG images at scale on Snowflake.

## Overview

This demo showcases Snowflake's `run_batch` API for **distributed GPU inference** of medical images. A single API call processes hundreds of ECG images through a 4-billion parameter vision-language model.

| Feature | Benefit |
|---------|---------|
| **Single API call** | `run_batch()` handles distribution, scheduling, and scaling |
| **GPU acceleration** | Automatic provisioning via Snowpark Container Services |
| **Multi-modal** | Process images + text prompts together natively |
| **Enterprise-ready** | Data never leaves Snowflake's secure environment |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ECG Images     │────▶│  Snowflake       │────▶│  Classification │
│  (928 images)   │     │  run_batch API   │     │  Results        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  GPU Compute     │
                        │  (NVIDIA A10G)   │
                        │  + MedGemma 4B   │
                        └──────────────────┘
```

## Prerequisites

1. **Snowflake Account** with:
   - ACCOUNTADMIN privileges (or equivalent)
   - Access to GPU compute pools (GPU_NV_M instance family)

2. **Kaggle Account** (free):
   - Get API key from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token

3. **MedGemma Model Access**:
   - Accept terms at [kaggle.com/models/google/medgemma](https://www.kaggle.com/models/google/medgemma)

## Quick Start

### 1. Run Setup SQL

Execute `setup.sql` in Snowflake to create required resources:

```sql
-- Creates:
-- - Database: MEDGEMMA_DEMO
-- - Stages: ECG_STAGE, ECG_BATCH_OUTPUT_STAGE
-- - Compute Pool: MEDGEMMA_COMPUTE_POOL (GPU_NV_M: 4x NVIDIA A10G)
-- - Network rules for external access
```

### 2. Register MedGemma Model

Before running the notebook, register the MedGemma model in Snowflake's Model Registry. This requires downloading from Kaggle and registering with vLLM inference engine support.

### 3. Run the Notebook

Open `medgemma_ecg_snowsight.ipynb` in Snowsight and execute cells sequentially:

1. **Setup** - Install dependencies and configure Kaggle API
2. **Download** - Fetch ECG dataset from Kaggle (928 images)
3. **Upload** - Stage images in Snowflake
4. **Inference** - Run distributed batch inference
5. **Analyze** - Parse and visualize classification results

## ECG Classification Categories

The model classifies ECGs into four categories:

| Category | Description |
|----------|-------------|
| `NORMAL` | Regular sinus rhythm, no abnormalities |
| `ABNORMAL_HEARTBEAT` | Irregular rhythm, ectopic beats, arrhythmia |
| `MYOCARDIAL_INFARCTION` | Acute MI signs (ST elevation, hyperacute T waves) |
| `POST_MI` | Old/healed MI (pathological Q waves) |

## Key Code: The `run_batch` Call

```python
from snowflake.ml.model import JobSpec, OutputSpec, SaveMode, InputSpec
from snowflake.ml.model.inference_engine import InferenceEngine

job = mv.run_batch(
    compute_pool="MEDGEMMA_COMPUTE_POOL",
    X=input_df,  # DataFrame with MESSAGES column
    input_spec=InputSpec(params={"temperature": 0.2, "max_tokens": 1024}),
    output_spec=OutputSpec(stage_location=output_location, mode=SaveMode.OVERWRITE),
    job_spec=JobSpec(gpu_requests="1"),
    inference_engine_options={
        "engine": InferenceEngine.VLLM,
        "engine_args_override": [
            "--max-model-len=7048",
            "--gpu-memory-utilization=0.9",
        ]
    }
)
```

## Performance

### Single Node (GPU_NV_M: 4x NVIDIA A10G)

| Images | Inference Time | Throughput | Avg/Image |
|--------|----------------|------------|-----------|
| 100 | ~60s | ~6,000 img/hr | 0.6s |
| 200 | ~147s | ~4,900 img/hr | 0.73s |
| 500 | ~330s | ~5,400 img/hr | 0.67s |

*Note: Startup overhead is ~2-3 minutes (model loading, vLLM warmup). Throughput measured during active inference.*

### Multi-Node (2x GPU_NV_M, replicas=2)

| Images | Inference Time | Throughput | Notes |
|--------|----------------|------------|-------|
| 100* | ~30s | ~6,000 img/hr | *No scaling - see note below |

*\*100 images: Node 2 initialized ~1.5 min after Node 1. With only ~30s of inference, Node 1 completed before Node 2 was ready. Multi-node scaling requires larger batches where inference time exceeds the initialization stagger.*

**Token usage per image**: ~1,100-1,220 tokens (836 prompt + 300-385 completion)

## Files

| File | Description |
|------|-------------|
| `setup.sql` | SQL script to create database, stages, and compute pool |
| `medgemma_ecg_snowsight.ipynb` | Main notebook for Snowsight |

## Resources

- [Snowflake ML Model Registry](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview)
- [Snowpark Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)
- [MedGemma on Kaggle](https://www.kaggle.com/models/google/medgemma)
- [ECG Dataset](https://www.kaggle.com/datasets/evilspirit05/ecg-analysis)

## License

This demo is provided for educational purposes. MedGemma is subject to Google's model terms. The ECG dataset is from Kaggle with its own license terms.
