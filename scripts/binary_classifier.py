# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import yaml

import typer
import os
from pathlib import Path
import shutil
import glob
from tqdm import tqdm
from typing import List
import json

from sigma_chan_network.data_structure.configrators import TrainConfig
from sigma_chan_network.components.models import SigmaChanCNN
from sigma_chan_network.io import S3Storage
from sigma_chan_network.components.operators import download_data

app = typer.Typer()
data_dir = str(Path(os.path.abspath(__file__)).parent.parent)

def create_image_generator(image_dir_path: str, data_flag: str,image_size: int = 224, batch_size: int = 32, shuffle: bool = True):

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    data_transform = {
        "train": torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]),
        "val": torchvision.transforms.Compose([
            #torchvision.transforms.RandomResizeCrop(image_size, scale=(0.5, 1.0)),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.Resize(image_size),
            
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]),
    }

    dataset = torchvision.datasets.ImageFolder(root=image_dir_path, transform=data_transform[data_flag])

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader

@app.command()
def train_local(job_id: str, parameters_path: str):
    with open(parameters_path, "r") as f_in:
        parameters_dict = yaml.safe_load(f_in)

    parameters_str = json.dumps(parameters_dict)
    print("parameters string: ", parameters_dict)

    train(job_id, parameters_str)


@app.command()
def train(job_id: str, parameters_str: str):
    print("Do something with cnn")
    print("Pamameters configuring:")
    parameters_dict = json.loads(parameters_str)
    config = TrainConfig(**parameters_dict)
    print(">> config: ", config)

    print("Downloading data from the bucket")
    s3 = S3Storage(config.cloud_storage)
    download_data(s3, config.local_storage)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("is cuda available? ", device)
    print("Configuring the model")
    model = SigmaChanCNN(config.model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1.0E-04)

    losses = []
    accs = []
    val_losses = []
    val_accs = []

    print("Creating image generator")
    train_dataloader = create_image_generator(os.path.join(data_dir, "data/train"), "train")
    val_dataloader = create_image_generator(os.path.join(data_dir, "data/val"), "val")

    print("Train")
    #for epoch in range(n_epoch):
    for epoch in range(config.n_epoch):
        running_loss = 0.0
        running_acc = 0.0
        print(f">> {epoch} th epoch training.")
        
        with tqdm(train_dataloader, total=len(train_dataloader)) as progressbar_loss:

            for imgs, labels in progressbar_loss:
            #for imgs, labels in train_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, labels)
                loss.backward()
                running_loss += loss.item()
                pred = torch.argmax(output, dim = 1)
                running_acc += torch.mean(pred.eq(labels).float())
                optimizer.step()
            
            running_loss /= len(train_dataloader)
            running_acc /= len(train_dataloader)
            print(f">>   running loss: {running_loss}, {type(running_loss)}")
            losses.append(running_loss)
            accs.append(running_acc)

        ### validation
        print(f">> {epoch} th epoch validation.")
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_imgs, val_labels in val_dataloader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            optimizer.zero_grad()
            val_output = model(val_imgs)
            val_loss = criterion(val_output, val_labels)
            val_loss.backward()
            val_running_loss += val_loss.item()
            val_pred = torch.argmax(val_output, dim = 1)
            val_running_acc += torch.mean(val_pred.eq(val_labels).float())
        
        val_running_loss /= len(val_dataloader)
        val_running_acc /= len(val_dataloader)
        print(f">>   val running loss: {val_running_loss}, {type(val_running_loss)}, {type(val_losses)}")
        val_losses.append(val_running_loss)
        val_accs.append(val_running_acc)
        print(f">> {epoch} th epoch: \n>>  train loss: {running_loss}, train acc: {running_acc}\n  >>  val loss: {val_running_loss}, val acc: {val_running_acc}")

        model_path = os.path.join(data_dir, "data/models", job_id, "{:04}.pth".format(epoch))
        print(f">> model_path: {model_path}")
        
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        
        if epoch == 0:
            __save_model(epoch, model, optimizer, val_losses, model_path, config, s3)
        elif epoch > 0:
            if val_losses[-1] == min(val_losses):
                __save_model(epoch, model, optimizer, val_losses, model_path, config, s3)

                    
def __save_model(epoch: int, model: SigmaChanCNN, optimizer: optim, val_losses: List[float], model_path: str, config: TrainConfig, s3: S3Storage):
    torch.save({"epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": val_losses[-1]}, model_path)
    model_file_name = str(os.path.join(*model_path.split("/")[-3:]))        
    model_path_in_bucket = os.path.join(config.local_storage.dir_name, model_file_name)
    print(">> save model", model_path, os.path.exists(model_path))
    print(">> uploaded model", model_path_in_bucket)
    s3.upload_file(model_path, model_path_in_bucket)
    
    

@app.command()
def prepare_dataset(weight_val: float = 0.3, weight_test: float = 0.1):
    print("Preparing dataset")
    print(">> root dir: ", data_dir)
    if weight_val + weight_test > 1.0:
        raise ValueError(f"weight of validation and test exceeded 1, weight_val: {weight_val}, weight_test: {weight_test}")
    
    ### kill exisiting data
    print("path to kill: ", os.path.join(data_dir, "data/train"), os.path.join(data_dir, "data/val"))
    shutil.rmtree(os.path.join(data_dir, "data/train"), ignore_errors=True)
    shutil.rmtree(os.path.join(data_dir, "data/val"), ignore_errors=True)
    shutil.rmtree(os.path.join(data_dir, "data/test"), ignore_errors=True)
    
    ### get oridinal data list and distribution list
    data_dict = {}
    data_dict["train"] = {}
    data_dict["val"] = {}
    data_dict["test"] = {}

    fav_list = list(set(glob.glob(f"{data_dir}/data/original/fav/*.*", recursive=True)))
    n_fav_list = len(fav_list)
    data_dict["train"]["fav"] = fav_list[: int(n_fav_list * (1.0 - weight_test - weight_val))]
    data_dict["val"]["fav"] = fav_list[int(n_fav_list * (1.0 - weight_test - weight_val)): int(n_fav_list * (1.0 - weight_test))]
    data_dict["test"]["fav"] = fav_list[int(n_fav_list * (1.0 - weight_test)):]

    ng_list = list(set(glob.glob(f"{data_dir}/data/original/ng/*.*", recursive=True)))
    n_ng_list = len(ng_list)
    data_dict["train"]["ng"] = ng_list[: int(n_ng_list * (1.0 - weight_test - weight_val))]
    data_dict["val"]["ng"] = ng_list[int(n_ng_list * (1.0 - weight_test - weight_val)): int(n_ng_list * (1.0 - weight_test))]
    data_dict["test"]["ng"] = ng_list[int(n_ng_list * (1.0 - weight_test)):]
    
    ### distribute
    print(">> Distributing files")
    dist_list = ["train", "val", "test"]
    class_list = ["fav", "ng"]
    for dist_item in dist_list:
        for class_label in class_list:
            destination_dir = os.path.join(data_dir, f"data/{dist_item}/{class_label}")
            print(f">>  working on {destination_dir}")
            os.makedirs(destination_dir)
            for item in data_dict[dist_item][class_label]:
                shutil.copy(item, destination_dir)
                
        
    

if __name__ == "__main__":
    app()