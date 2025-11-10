#!/usr/bin/env python3
"""
Simple vLLM test with Eagle speculative decoding.
"""

import os
import pandas as pd
from vllm import LLM, SamplingParams
from prometheus_client import start_http_server

# Environment variables
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("HF_HOME", "/work/hdd/bcjw/xsong3")

# Config
MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen3-4B-Base")
DATASET_PATH = os.environ.get("TEST_DATASET_PATH", "/u/xsong3/data/dapo_math/train.parquet")
DATASET_PROMPT_KEY = os.environ.get("TEST_DATASET_PROMPT_KEY", "prompt")
EAGLE_DRAFT_MODEL = os.environ.get("TEST_EAGLE_DRAFT_MODEL", "AngelSlim/Qwen3-4B_eagle3")
EAGLE_NUM_SPEC_TOKENS = int(os.environ.get("TEST_EAGLE_NUM_SPEC_TOKENS", "8"))
BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "10"))

# vLLM config parameters (matching vllm_rollout_spmd.py)
TENSOR_PARALLEL_SIZE = int(os.environ.get("TEST_TENSOR_PARALLEL_SIZE", "1"))
MAX_PROMPT_LENGTH = int(os.environ.get("TEST_MAX_PROMPT_LENGTH", "1024"))
MAX_RESPONSE_LENGTH = int(os.environ.get("TEST_MAX_RESPONSE_LENGTH", "4096"))
MAX_MODEL_LEN = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("TEST_MAX_NUM_BATCHED_TOKENS", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("TEST_GPU_MEMORY_UTIL", "0.7"))
ENABLE_CHUNKED_PREFILL = os.environ.get("TEST_ENABLE_CHUNKED_PREFILL", "False").lower() == "true"
FREE_CACHE_ENGINE = os.environ.get("TEST_FREE_CACHE_ENGINE", "False").lower() == "true"

# 1. Start Prometheus server
start_http_server(9108)
print("Prometheus server started on port 9108")

# 2. Initialize LLM with Eagle speculative decoding (matching vllm_rollout_spmd.py)
print("Initializing vLLM with Eagle speculative decoding...")
llm = LLM(
    model=MODEL_PATH,
    enable_sleep_mode=FREE_CACHE_ENGINE,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    dtype="bfloat16",
    enforce_eager=False,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    disable_custom_all_reduce=True,
    skip_tokenizer_init=False,
    max_model_len=MAX_MODEL_LEN,
    load_format="auto",
    disable_log_stats=False,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
    enable_prefix_caching=True,
    trust_remote_code=True,
    seed=0,
    speculative_config={
        "method": "eagle3",
        "model": EAGLE_DRAFT_MODEL,
        "num_speculative_tokens": EAGLE_NUM_SPEC_TOKENS,
        "draft_tensor_parallel_size": 1,
    },
)
print("vLLM initialized")

# 3. Load dataset and generate
print(f"Loading dataset from {DATASET_PATH}")
df = pd.read_parquet(DATASET_PATH)
prompts = df[DATASET_PROMPT_KEY].dropna().astype(str).tolist()
prompts = [p.strip() for p in prompts if p.strip()]
print(f"Loaded {len(prompts)} prompts")

# Sampling params
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# Generate in batches
print("Starting generation...")
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    print(f"Batch {i//BATCH_SIZE + 1}: Processing {len(batch_prompts)} prompts")
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=False)
    print(f"Batch {i//BATCH_SIZE + 1}: Completed")

print("Generation completed")
