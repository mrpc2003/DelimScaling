#!/usr/bin/env bash
# Fixed OpenResearch run contract for both baseline and delimiter-scaling nodes.
set -euo pipefail

source reproduction/variant.env

CONDA_BIN="/opt/conda/bin/conda"
CACHE_ROOT="${HOME}/.cache/delimscaling-reproduction"
ENV_PREFIX="${CACHE_ROOT}/py310-torch271-cu128"
OUTPUT_DIR="${PWD}/reproduction_outputs/${VARIANT_NAME}"

mkdir -p "${CACHE_ROOT}" "${OUTPUT_DIR}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --yes --prefix "${ENV_PREFIX}" python=3.10 pip
fi

PYTHON="${ENV_PREFIX}/bin/python"

# The versions below deliberately avoid the system Python 3.8 / PyTorch 1.9 /
# CUDA 11.1 stack.  Both variants share this persistent, isolated environment.
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
"${PYTHON}" -m pip install \
  accelerate==1.8.1 datasets==3.6.0 decord==0.6.0 evaluate==0.4.3 loguru==0.7.3 \
  hf-transfer==0.1.9 matplotlib==3.9.4 openai==1.99.9 pandas==2.2.3 sacrebleu==2.5.1 seaborn==0.13.2 sqlitedict==2.1.0 \
  tenacity==9.1.2 transformers==4.53.1 qwen-vl-utils==0.0.14 pytablewriter==1.2.1

# The repository carries a patched Transformers tree. PYTHONPATH ensures this
# run's committed tree is used while the dependency environment stays identical.
export PYTHONPATH="${PWD}/transformers/src:${PWD}/qwen-vl-utils/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TOKENIZERS_PARALLELISM=false
# Diagnostic children may synchronize CUDA to expose the true failing operation.
# Normal baseline and scaled runs leave this at 0.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

"${PYTHON}" - <<'PY'
import torch
import transformers
print(f"REPRO_ENV python={__import__('sys').version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}")
print(f"REPRO_ENV cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA was not available in the isolated environment"
assert torch.cuda.device_count() >= 2, "Expected the inspected two-GPU local machine"
PY

# SDPA is a controlled deviation from the upstream README's FlashAttention 2
# path.  The local CUDA 11.2 nvcc cannot build Blackwell sm_120 FlashAttention,
# so the committed Qwen patch restores the requested SDPA class for both vision
# and language attention. It is held fixed across baseline and scaled variants.
LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
fi
"${PYTHON}" -m accelerate.commands.launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=cuda,attn_implementation=sdpa" \
  --tasks muirbench \
  --batch_size 1 \
  --select_layer "${SELECT_LAYERS}" \
  --delim_scaling "${DELIM_SCALING}" \
  --scale "${SCALE}" \
  --seed 0,1234,1234,1234 \
  "${LIMIT_ARGS[@]}" \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  --verbosity INFO

RESULT_JSON="$(find "${OUTPUT_DIR}" -name '*_results.json' -type f -print | sort | tail -n 1)"
test -n "${RESULT_JSON}"
"${PYTHON}" reproduction/summarize_results.py \
  --results "${RESULT_JSON}" \
  --variant "${VARIANT_NAME}" \
  --scaling "${DELIM_SCALING}" \
  --scale "${SCALE}" \
  --layers "${SELECT_LAYERS}" \
  --attention sdpa \
  --limit "${EVAL_LIMIT}"
