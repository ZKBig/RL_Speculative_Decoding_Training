#!/bin/bash

PROJECT_DIR=/u/xsong3/myproj/rl_sd/Spec-RL
MODEL_PATH=Qwen/Qwen3-1.7B-Base
SAVE_PATH=/work/hdd/bcjw/xsong3/ckpt/1.7B-grpo-lenience-0.5
PROJECT_NAME=verl-grpo-lenience-0.5-train

# for vllm 0.9.1: uncomment to fix oov issue
bash ${PROJECT_DIR}/training_scripts/fix_vllm_oov.sh ${PROJECT_DIR}

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}


bash ${PROJECT_DIR}/training_scripts/train_grpo-spec-sampling.sh \
    --dataset_name deepmath \
    --train_file_name train_sample_6144 \
    --model_name Qwen3-1.7B-Base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.8 \
    --rollout_tp 2 \
    --total_epochs 20 \
    --total_steps 100 \
    --rollout_name vllm \
    --save_freq 10 \
    --spec_decoding True \
    --bias 0.5 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}