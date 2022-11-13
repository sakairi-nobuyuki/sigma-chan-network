# coding: utf-8

from typing import Optional

from pydantic import BaseModel


class StorageConfig(BaseModel):
    type: str = "s3"
    project_name: Optional[str] = ""
    bucket_name: str = "getter-robo"
    endpoint_url: str = "http://192.168.0.16:9000"
    access_id: str = "sigma-chan"
    access_key: str = "sigma-chan-dayo"
    region_name: str = ""
    dir_name: str = ""
