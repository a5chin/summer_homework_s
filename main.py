from pathlib import Path

import pyheif
from PIL import Image

root = Path("/Users/a5chin/Desktop/image")
to = Path("/Users/a5chin/Python/task_s/assets/data")

for path in root.glob("**/*.HEIC"):
    d = to / path.parent.name
    heif = pyheif.read(path)

    img = Image.frombytes(
        heif.mode,
        heif.size,
        heif.data,
        "raw",
        heif.mode,
        heif.stride,
    )
    print(d / f"{path.stem}.png")
    img.save(d / f"{path.stem}.png")
