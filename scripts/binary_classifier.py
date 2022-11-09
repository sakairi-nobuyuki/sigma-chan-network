# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import typer
import os
from pathlib import Path
import shutil
import glob

app = typer.Typer()
data_dir = str(Path(os.path.abspath(__file__)).parent.parent)


@app.command()
def train(n_epoch: int = 50):
    print("do something with cnn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>  cuda is available: ", device)

    model = SigmaChanCnn(device)
    model.to(device)

    criterion = nn.CrossEntrophyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1.0E-04)

    losses = []
    accs = []
    val_losses = []
    val_accs = []

    train_dataloader = []
    val_dataloader = []

    for epoch in range(n_epoch):
        running_loss = 0.0
        running_acc = 0.0
        
        for img, labels in train_dataloader:
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
        losses.append(running_loss)
        accs.append(running_acc)

        ### validation
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_imgs, val_labels in val_dataloader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            optimizer.zero_grad()
            val_output = model(imgs)
            val_loss = criterion(val_output, val_labels)
            val_loss.backward()
            val_running_loss += val_loss.item()
            val_pred = torch.argmax(val_output, dim = 1)
            val_running_acc += torch.mean(pred.eq(val_labels).float())
        val_running_loss /= len(val_dataloader)
        val_running_acc /= len(val_dataloader)
        val_loss.append(val_running_loss)
        accs.append(running_acc)
        print(f"{epoch} th epoch: \n>>  train loss: {running_loss}, train acc: {running_acc}\n  >>  val loss: {val_running_loss}, val acc: {val_running_acc}")


class SigmaChanCnn(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channnels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channnels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channnels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channnels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features = 4 * 4 * 128, out_features=num_classes)
    def forward(self, x) -> None:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


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
    #fav_train_list = fav_list[: int(n_fav_list * (1.0 - weight_test - weight_val))]
    data_dict["train"]["fav"] = fav_list[: int(n_fav_list * (1.0 - weight_test - weight_val))]
    #fav_val_list = fav_list[int(n_fav_list * (1.0 - weight_test - weight_val)): int(n_fav_list * (1.0 - weight_test))]
    data_dict["val"]["fav"] = fav_list[int(n_fav_list * (1.0 - weight_test - weight_val)): int(n_fav_list * (1.0 - weight_test))]
    #fav_test_list = fav_list[int(n_fav_list * (1.0 - weight_test)):]
    data_dict["test"]["fav"] = fav_list[int(n_fav_list * (1.0 - weight_test)):]

    ng_list = list(set(glob.glob(f"{data_dir}/data/original/ng/*.*", recursive=True)))
    n_ng_list = len(ng_list)
    #ng_train_list = fav_list[: int(n_ng_list * (1.0 - weight_test - weight_val))]
    data_dict["train"]["ng"] = fav_list[: int(n_ng_list * (1.0 - weight_test - weight_val))]
    #ng_val_list = fav_list[int(n_ng_list * (1.0 - weight_test - weight_val)): int(n_ng_list * (1.0 - weight_test))]
    data_dict["val"]["ng"] = fav_list[int(n_ng_list * (1.0 - weight_test - weight_val)): int(n_ng_list * (1.0 - weight_test))]
    #ng_test_list = fav_list[int(n_ng_list * (1.0 - weight_test)):]
    data_dict["test"]["ng"] = fav_list[int(n_ng_list * (1.0 - weight_test)):]
    
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