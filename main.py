# coding: utf-8

from sigma_chan_network.pipelines import trainer
import typer
import os
from pathlib import Path

app = typer.Typer()
data_dir = str(Path(os.path.abspath(__file__)).parent.parent)

@app.command()
def train_local(job_id: str, parameters_path: str) -> None:

    trainer.train_local(job_id, parameters_path)

@app.command()                
def train(job_id: str, parameters_str: str) -> None:

    trainer.train(job_id, parameters_str)
    

if __name__ == "__main__":
    app()