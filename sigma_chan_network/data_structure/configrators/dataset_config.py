# conding: utf-8

from typing import Optional

from pydantic import BaseModel, PositiveFloat, root_validator


class DatasetConfig(BaseModel):
    weight_val: Optional[PositiveFloat] = 0.2
    weight_test: Optional[PositiveFloat] = 0.1
    weight_train: Optional[PositiveFloat]
    data_path: Optional[str] = "data"

    @root_validator
    def __validata_train_weight(cls, values):
        if values["weight_test"] + values["weight_val"] > 1.0:
            raise ValueError(
                f"The sum of weight of validation and test should not exceed 1: val: {values['weight_val']}, test: {values['weight_test']}"
            )

        values["weight_train"] = 1.0 - values["weight_test"] - values["weight_val"]

        return values
