#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12


if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/algorithms/SVO" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/../algorithms/SVO" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SVO_DIR="${PROJECT_DIR}/algorithms/SVO"

echo "Project directory: ${PROJECT_DIR}"
echo "SVO directory: ${SVO_DIR}"

source activate jax

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_DIR="${PROJECT_DIR}"

cd "${PROJECT_DIR}"

COMMON_OVERRIDES=(
  "TUNE=False"
  "WANDB_MODE=${WANDB_MODE:-online}"
  "SEED=${SEED:-30}"
  "NUM_SEEDS=${NUM_SEEDS:-1}"
  "ENV_NAME=coop_mining"
  "ENV_KWARGS.num_agents=2"
  "ENV_KWARGS.shared_rewards=False"
  "+ENV_KWARGS.num_outer_steps=1"
  "+ENV_KWARGS.max_miners=${MAX_MINERS:-4}"
  "+ENV_KWARGS.min_gold_miners=${MIN_GOLD_MINERS:-2}"
  "+ENV_KWARGS.mining_range=${MINING_RANGE:-3}"
  "+ENV_KWARGS.reward_iron=${REWARD_IRON:-1.0}"
  "+ENV_KWARGS.reward_gold=${REWARD_GOLD:-8.0}"
  "+ENV_KWARGS.gold_mining_window=${GOLD_MINING_WINDOW:-3}"
  "+ENV_KWARGS.regrowth_prob_iron=${REGROWTH_PROB_IRON:-0.0004}"
  "+ENV_KWARGS.regrowth_prob_gold=${REGROWTH_PROB_GOLD:-0.00016}"
  "ENV_KWARGS.cnn=True"
  "ENV_KWARGS.jit=True"
  "ENV_KWARGS.svo=True"
  "ENV_KWARGS.svo_target_agents=[0,1]"
  "ENV_KWARGS.svo_w=${SVO_W:-0.5}"
  "ENV_KWARGS.svo_ideal_angle_degrees=${SVO_IDEAL_ANGLE_DEGREES:-90}"
  "PARAMETER_SHARING=True"
  "TOM_AUX_HIDDEN=True"
  "TOM_STOP_GRAD_PARTNER=${TOM_STOP_GRAD_PARTNER:-True}"
)

if [[ -n "${NUM_ENVS:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_ENVS=${NUM_ENVS}")
fi

if [[ -n "${NUM_STEPS:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_STEPS=${NUM_STEPS}")
fi

if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
  COMMON_OVERRIDES+=("TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}")
fi

# Paired recurrent PPO minibatches keep both agents from each env together, so
# NUM_MINIBATCHES must divide NUM_ENVS.
if [[ -n "${NUM_MINIBATCHES:-}" ]]; then
  COMMON_OVERRIDES+=("NUM_MINIBATCHES=${NUM_MINIBATCHES}")
fi

if [[ -n "${LR:-}" ]]; then
  COMMON_OVERRIDES+=("LR=${LR}")
fi

echo "Training Coop Mining SVO RNN with ToM partner hidden input"
python "${SVO_DIR}/svo_rnn.py" "${COMMON_OVERRIDES[@]}"
