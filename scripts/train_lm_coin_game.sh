#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12


if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/algorithms/IPPO" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/../algorithms/IPPO" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
IPPO_DIR="${PROJECT_DIR}/algorithms/IPPO"

echo "Project directory: ${PROJECT_DIR}"
echo "IPPO directory: ${IPPO_DIR}"

source activate jax

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_DIR="${PROJECT_DIR}"

cd "${PROJECT_DIR}"

SEED_VALUE="${SEED:-30}"
DEFAULT_PRETRAINED_PATH="${PROJECT_DIR}/checkpoints/individual/coin_game_single_rnn_seed${SEED_VALUE}.pkl"
PRETRAINED_PATH="${PRETRAINED_PARAMS_PATH:-${DEFAULT_PRETRAINED_PATH}}"

COMMON_OVERRIDES=(
  "TUNE=False"
  "WANDB_MODE=${WANDB_MODE:-online}"
  "SEED=${SEED_VALUE}"
  "NUM_SEEDS=${NUM_SEEDS:-5}"
  "ENV_NAME=coin_game"
  "ENV_KWARGS.num_agents=2"
  "ENV_KWARGS.shared_rewards=False"
  "ENV_KWARGS.cnn=True"
  "ENV_KWARGS.jit=True"
  "PARAMETER_SHARING=True"
  "PRETRAINED_PARAMS_PATH=${PRETRAINED_PATH}"
  "TRANSFORM_TASK=coin_game"
  "PERSPECTIVE_TRANSFORM=${PERSPECTIVE_TRANSFORM:-True}"
  "USE_SELF_PRED=${USE_SELF_PRED:-True}"
  "FINETUNE_SELF_STREAM=${FINETUNE_SELF_STREAM:-True}"
  "FINETUNE_OTHER_STREAM=${FINETUNE_OTHER_STREAM:-False}"
)

if [[ -n "${RUN_NAME:-}" ]]; then
  COMMON_OVERRIDES+=("RUN_NAME=${RUN_NAME}")
fi

if [[ -n "${NUM_ENVS:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_ENVS=${NUM_ENVS}")
fi

if [[ -n "${NUM_STEPS:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_STEPS=${NUM_STEPS}")
fi

if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  COMMON_OVERRIDES+=("TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}")
fi

# Recurrent PPO minibatches complete actor sequences, so NUM_MINIBATCHES must
# divide NUM_ENVS * 2 for this two-agent Coin Game run.
if [[ -n "${NUM_MINIBATCHES:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_MINIBATCHES=${NUM_MINIBATCHES}")
fi

if [[ -n "${LR:-}" ]]; then
  COMMON_OVERRIDES+=("LR=${LR}")
fi

echo "Training Coin Game LM with recurrent IPPO"
echo "Pretrained single-agent params: ${PRETRAINED_PATH}"
python "${IPPO_DIR}/ippo_lm.py" "${COMMON_OVERRIDES[@]}"
