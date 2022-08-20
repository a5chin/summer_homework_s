from torch import nn, optim

from .dataset import get_loader
from .transform import get_transforms


class Trainer:
    def __init__(
        self,
        cfg,
    ) -> None:
        self.cfg = cfg
        self.transform = get_transforms()
        self.train_loader = get_loader(
            root=cfg.root,
            batch_size=cfg.batch_size,
            transform=self.transform["train"],
        )

    def fit(self):
        for epoch in range(self.cfg.epochs):
            for images, target in self.train_loader:
                print(images, target)
                exit(0)
