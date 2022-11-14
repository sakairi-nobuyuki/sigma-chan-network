# coding: utf-8

import glob
import os
import shutil

from sigma_chan_network.data_structure.configrators import DatasetConfig, StorageConfig


def create_fav_ng_dataset(local_storage_config: StorageConfig, dataset_config: DatasetConfig):
    """Distributing a data to a dataset by distributing from the original data to train, validation and test data.
    The directory configuration is,
    
    data --- train_data --- original --- fav
                         |            |- ng
                         |
                         |- dataset --- train --- fav
                                     |         |- ng
                                     |
                                     |- val --- fav
                                     |       |- ng
                                     |
                                     |- test --- fav
                                              |- ng

    Args:
        local_storage_config (StorageConfig): local storage configuration
        dataset_config (DatasetConfig): dataset configuration

    Raises:
        ValueError: The inconsisntency of the rate of the amount of each train, validation and test data 
    """
    print("Preparing dataset")
    print(">> root dir: ", local_storage_config.dir_name)
    if dataset_config.weight_val + dataset_config.weight_test > 1.0:
        raise ValueError(
            f"weight of validation and test exceeded 1, weight_val: {dataset_config.weight_val}, weight_test: {dataset_config.weight_test}"
            )
    
    ### kill exisiting data
    print("path to kill: ", os.path.join(local_storage_config.dir_name, "train_data/dataset"))
    shutil.rmtree(os.path.join(local_storage_config.dir_name, "train_data/dataset"), ignore_errors=True)
    
    ### get oridinal data list and distribution list
    data_dict = {}
    data_dict["train"] = {}
    data_dict["val"] = {}
    data_dict["test"] = {}

    fav_list = list(
        set(glob.glob(f"{local_storage_config.dir_name}/train_data/original/fav/*.*", recursive=True))
        )
    n_fav_list = len(fav_list)
    data_dict["train"]["fav"] = fav_list[: int(n_fav_list * (1.0 - dataset_config.weight_test - dataset_config.weight_val))]
    data_dict["val"]["fav"] = fav_list[
        int(n_fav_list * (1.0 - dataset_config.weight_test - dataset_config.weight_val)): int(n_fav_list * (1.0 - dataset_config.weight_test))]
    data_dict["test"]["fav"] = fav_list[int(n_fav_list * (1.0 - dataset_config.weight_test)):]

    ng_list = list(set(glob.glob(f"{local_storage_config.dir_name}/train_data/original/ng/*.*", recursive=True)))
    n_ng_list = len(ng_list)
    data_dict["train"]["ng"] = ng_list[: int(n_ng_list * (1.0 - dataset_config.weight_test - dataset_config.weight_val))]
    data_dict["val"]["ng"] = ng_list[
        int(n_ng_list * (1.0 - dataset_config.weight_test - dataset_config.weight_val)): int(n_ng_list * (1.0 - dataset_config.weight_test))]
    data_dict["test"]["ng"] = ng_list[int(n_ng_list * (1.0 - dataset_config.weight_test)):]
    
    ### distribute
    print(">> Distributing files")
    dist_list = ["train", "val", "test"]
    class_list = ["fav", "ng"]
    for dist_item in dist_list:
        for class_label in class_list:
            destination_dir = os.path.join(local_storage_config.dir_name, f"train_data/dataset/{dist_item}/{class_label}")
            print(f">>  working on {destination_dir}")
            os.makedirs(destination_dir)
            for item in data_dict[dist_item][class_label]:
                shutil.copy(item, destination_dir)
