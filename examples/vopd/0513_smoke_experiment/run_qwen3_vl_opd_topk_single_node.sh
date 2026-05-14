#!/usr/bin/env bash
# Single-node smoke test for Qwen3-VL top-k on-policy distillation.
#
# This mirrors run_qwen3_vl_opd_smoke_single_node.sh, but switches the
# distillation objective from sampled-token k1 to forward_kl_topk.

set -xeuo pipefail

# ---- user-adjustable paths ----
STUDENT_MODEL=${STUDENT_MODEL:-../models/qwen3-vl-8b}
TEACHER_MODEL=${TEACHER_MODEL:-../models/qwen3-vl-32b}

TRAIN_FILE=${TRAIN_FILE:-../preprocessed_dataset/vopd_smoke/train.parquet}
VAL_FILE=${VAL_FILE:-../preprocessed_dataset/vopd_smoke/val.parquet}

# ---- single-node 8-GPU defaults ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-forward_kl_topk}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-False}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6000}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.7}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-100000}
TEST_FREQ=${TEST_FREQ:-3}

PROJECT_NAME=${PROJECT_NAME:-vopd}
DATE=${DATE:-$(date +%m%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}_qwen3-vl_opd_topk_smoke_single_node}
LOGGER=${LOGGER:-'["console","wandb"]'}
EXP_NAME=${EXP_NAME:-${EXPERIMENT_NAME}}
ROLLOUT_SAVE_PATH="./rollouts_saved/${EXP_NAME}"

max_num_tokens=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}/train"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

REWARD=(
    reward.custom_reward_function.path=examples/vopd/reward_function.py
    reward.custom_reward_function.name=compute_score
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${DISTILLATION[@]}" \
    "$@"
