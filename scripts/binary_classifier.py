# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np

import typer
import os
from pathlib import Path
import shutil
import glob
from tqdm import tqdm
import datetime

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
def train(n_epoch: int = 50):
    print("do something with cnn")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>  cuda is available: ", device)
    print("Configuring the model")
    model = SigmaChanCnn(num_classes = 2)
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
    for epoch in range(n_epoch):
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

        model_path = os.path.join(data_dir, "models", datetime.datetime.now().strftime("%Y%m%d"), "{:04}.pth".format(epoch))
        print(f">> model_path: {model_path}")
        
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        
        if epoch == 0:
            torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_losses[-1]}, model_path)
        elif epoch > 0:
            if val_losses[-1] == min(val_losses):
                torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_losses[-1]}, model_path)
                    

class SigmaChanCnn(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        print(">> initializing model")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        print(">>   model layers: ", self.features)
        print(">> creating classifier")
        
        self.classifier = nn.Linear(in_features = 28 * 28 * 128,   out_features=num_classes)
        print(">>   classifier content: ", self.classifier)

    def forward(self, x) -> None:
        print("in forward: input: ", type(x))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        print("in forward: output: ", type(x))
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