#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IPPO_DIR="${PROJECT_DIR}/algorithms/IPPO"

echo "Project directory: ${PROJECT_DIR}"
echo "IPPO directory: ${IPPO_DIR}"

source activate jax

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_DIR="${PROJECT_DIR}"

cd "${PROJECT_DIR}"

COMMON_OVERRIDES=(
  "TUNE=False"
  "WANDB_MODE=${WANDB_MODE:-online}"
)

if [[ -n "${SEED:-}" ]]; then
  COMMON_OVERRIDES+=("SEED=${SEED}")
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
# divide NUM_ENVS * num_agents for each task.
if [[ -n "${NUM_MINIBATCHES:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_MINIBATCHES=${NUM_MINIBATCHES}")
fi

# echo "Training Cleanup Single with recurrent IPPO"
# python ippo_rnn_cleanup_single.py "${COMMON_OVERRIDES[@]}"

# echo "Training Harvest Single with recurrent IPPO"
# python ippo_rnn_harvest_single.py "${COMMON_OVERRIDES[@]}"

echo "Training Coin Game Single with recurrent IPPO"
python "${IPPO_DIR}/ippo_rnn_coin_game_single.py" "${COMMON_OVERRIDES[@]}"
