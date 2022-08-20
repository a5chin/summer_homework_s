from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class SummerDataset(ImageFolder):
    def __init__(
        self,
        root: str = "../assets/data",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform)
        self.root = root
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]

        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_dataset(
    root: str,
    transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    dataset = SummerDataset(root=root, transform=transform)

    return dataset


def get_loader(
    root: str, batch_size: int, transform: transforms.Compose
) -> Tuple[DataLoader, DataLoader]:
    dataset = get_dataset(root=root, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader
