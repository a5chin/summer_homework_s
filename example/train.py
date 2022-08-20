import argparse
import pathlib
import random
import sys

import torch
from torch import optim

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.as_posix()))

from summer import Trainer
from summer.dataset import get_dataset, get_loader
from summer.transform import get_transforms


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", default=16, type=int, help="plese set batch-size"
    )
    parser.add_argument(
        "--device", default="cuda:0", type=str, help="plese set device"
    )
    parser.add_argument(
        "--epochs", default=300, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--lr", default=0.05, type=float, help="initial (base) learning rate"
    )
    parser.add_argument(
        "-r",
        "--root",
        default="../assets/data",
        type=str,
        help="please set data root",
    )
    parser.add_argument(
        "--logdir", default="logs", type=str, help="please set logdir"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training"
    )

    return parser.parse_args()


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = make_parser()
    set_seed(args.seed)
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    trainer = Trainer(cfg=args)
    trainer.fit()


if __name__ == "__main__":
    main()
