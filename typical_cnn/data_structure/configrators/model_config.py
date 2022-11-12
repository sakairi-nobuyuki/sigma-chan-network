# coding: utf-8

from typing import Dict, Optional, Union

from pydantic import BaseModel, PositiveFloat, PositiveInt, root_validator


class ModelConfig(BaseModel):
    """Neural network config

    Args:
        BaseModel (_type_): pydantic

    """

    n_classes: PositiveInt
    n_channels_last_layer: PositiveInt = 128
    input_dim: Union[PositiveInt, Dict[str, PositiveInt]] = 224
    input_dim_x: Optional[PositiveInt] = None
    input_dim_y: Optional[PositiveInt] = None
    batch_size: PositiveInt = 32
    learning_rate: PositiveFloat = 0.001
    weight_decay: PositiveFloat = 1.0e-04

    @root_validator(pre=True)
    def __validate_input_dim(cls, values):
        """Validation for input dimension

        Args:
            values (_type_): entire values

        Returns:
            _type_: values
        """
        print("values: ", values)
        print("input dim: ", values["input_dim"])
        if isinstance(values["input_dim"], Dict):
            values["input_dim_x"] = values["input_dim"]["x"]
            values["input_dim_y"] = values["input_dim"]["y"]
        else:
            values["input_dim_x"] = values["input_dim"]
            values["input_dim_y"] = values["input_dim"]

        return values
