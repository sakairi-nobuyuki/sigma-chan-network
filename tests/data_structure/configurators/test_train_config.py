# coding: utf-8

from sigma_chan_network.data_structure.configrators import TrainConfig


class TestTrainConfig:
    """Testing train config"""

    def test_init(self, mock_train_config):
        """Test to load configuration"""
        config = TrainConfig(**mock_train_config)

        assert isinstance(config, TrainConfig)
        assert (
            config.local_storage.dir_name
            == mock_train_config["local_storage"]["dir_name"]
        )
