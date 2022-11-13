# coding: utf-8

import pytest


@pytest.fixture
def mock_model_config():
    """Model config mock for CNN"""
    return dict(n_classes=2, input_dim=224, batch_size=64, learning_rate=0.001, weight_decay=0.001)


@pytest.fixture
def mock_model_config_different_dims():
    """Model config mock with different x-y input dimension for CNN"""
    return dict(
        n_classes=2,
        input_dim=dict(x=224, y=256),
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.001,
    )


@pytest.fixture
def mock_dataset_config():
    """Dataset config mock"""
    return dict(wewight_val=0.2, weight_test=0.1, data_path="data")

@pytest.fixture
def mock_train_config():
    config_dict = {
        "cloud_storage": {
            "endpoint_url": "http://localhost:9000",
            "dir_name": "hoge/"
        }, 
        "local_storage": {
            "type": "local",
            "dir_name": "data/train_data/"
        },
        "model": {
            "n_classes": 2
        }, 
        "dataset": {}
    
    }

    return config_dict