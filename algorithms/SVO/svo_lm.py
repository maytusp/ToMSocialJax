"""SVO two-stream recurrent LM trainer with pretrained SocialJax encoders."""

import sys
from pathlib import Path

import hydra

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.IPPO.ippo_lm import main_from_config


@hydra.main(version_base=None, config_path="config", config_name="svo_lm")
def main(config):
    main_from_config(config, "svo_lm")


if __name__ == "__main__":
    main()
