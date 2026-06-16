import hydra

from ippo_rnn_single import main_from_config


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_coin_game_single")
def main(config):
    main_from_config(config, "ippo_rnn_coin_game_single")


if __name__ == "__main__":
    main()
