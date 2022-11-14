# coding: utf-8

import pytest

from sigma_chan_network.components.operators import create_fav_ng_dataset
from sigma_chan_network.data_structure.configrators import TrainConfig


@pytest.mark.skip
def test_create_fav_ng_dataset(mock_train_config):
    config = TrainConfig(**mock_train_config)
    config.local_storage.dir_name = "data"
    print(config)
    

    create_fav_ng_dataset(config.local_storage, config.dataset)

