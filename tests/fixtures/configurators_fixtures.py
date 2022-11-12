# coding: utf-8

import pytest


@pytest.fixture
def mock_model_config():
    return dict(n_classes=2, input_dim=224, batch_size=64, learning_rate=0.001, weight_decay=0.001)


@pytest.fixture
def mock_model_config_different_dims():
    return dict(
        n_classes=2,
        input_dim=dict(x=224, y=256),
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.001,
    )


@pytest.fixture
def mock_dataset_config():
    return dict(wewight_val=0.2, weight_test=0.1, data_path="data")
