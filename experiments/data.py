"""Datasets and training tasks for LHD's experiments.

Each dataset is downloaded once and cached as tensors in data/.
A task hands out shuffled training batches, scores the model on the
validation set, and holds the fixed probe batch LHD uses to count
dead units.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")
GTSRB_URL = ("https://huggingface.co/datasets/tanganke/gtsrb/resolve/main/"
             "data/{split}-00000-of-00001.parquet")
PROBE_SIZE = 256
EVAL_CHUNK = 2048


def _cached(name: str, build):
    """Load data/<name>.pt, building and saving it first if it is missing."""
    path = DATA_DIR / f"{name}.pt"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(build(), path)
    return torch.load(path)


def _download(url: str, path: Path) -> Path:
    """Fetch url to path unless it is already there."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def _build_gtsrb() -> dict:
    """Download GTSRB and shrink every image to 32x32, stored as uint8.

    Uses the Hugging Face copy of the torchvision splits (26,640
    training and 12,630 test images), since the original host often
    refuses connections.
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    result = {}
    for split in ("train", "test"):
        path = _download(GTSRB_URL.format(split=split), DATA_DIR / "raw" / "gtsrb_hf" / f"{split}.parquet")
        table = pq.read_table(path).to_pydict()
        images = []
        for item in table["image"]:
            with Image.open(io.BytesIO(item["bytes"])) as img:
                small = img.convert("RGB").resize((32, 32), Image.BILINEAR)
            images.append(np.asarray(small, dtype=np.uint8))
        result[f"{split}_x"] = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
        result[f"{split}_y"] = torch.tensor(table["label"], dtype=torch.long)
    return result


def _build_fashion() -> dict:
    """Download Fashion-MNIST as uint8 tensors."""
    from torchvision.datasets import FashionMNIST

    result = {}
    for split, train in (("train", True), ("test", False)):
        dataset = FashionMNIST(str(DATA_DIR / "raw"), train=train, download=True)
        result[f"{split}_x"] = dataset.data.clone()
        result[f"{split}_y"] = dataset.targets.clone()
    return result


def _build_shakespeare() -> dict:
    """Download Tiny Shakespeare and encode it one character per token."""
    text = urllib.request.urlopen(SHAKESPEARE_URL, timeout=60).read().decode("utf-8")
    chars = sorted(set(text))
    lookup = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([lookup[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(tokens))
    return {"train": tokens[:split], "test": tokens[split:], "vocab": len(chars)}


def _probe_indices(count: int) -> torch.Tensor:
    """The same validation examples every run, chosen once with seed 0."""
    generator = torch.Generator().manual_seed(0)
    return torch.randperm(count, generator=generator)[:PROBE_SIZE]


@dataclass
class ImageTask:
    """Classification on a fixed set of image tensors."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    classes: int
    batch_size: int

    def __post_init__(self):
        self.probe = self.val_x[_probe_indices(len(self.val_x))].numpy()

    def batches(self, generator: torch.Generator):
        """One epoch of shuffled training batches."""
        order = torch.randperm(len(self.train_x), generator=generator)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            yield self.train_x[idx], self.train_y[idx]

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    @staticmethod
    def hits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
        """Correct predictions and number of predictions."""
        return int((logits.argmax(-1) == targets).sum()), targets.numel()

    def evaluate(self, model) -> tuple[float, float]:
        """Validation accuracy and mean loss."""
        return _evaluate(model, self, self.val_x, self.val_y)


@dataclass
class TextTask:
    """Next-character prediction on one long token stream.

    An epoch is a fixed number of steps, each a batch of random
    windows, since the stream has no natural pass length.
    """

    train: torch.Tensor
    val: torch.Tensor
    vocab: int
    context: int
    batch_size: int
    steps: int
    val_windows: int = 512

    def __post_init__(self):
        generator = torch.Generator().manual_seed(1)
        starts = torch.randint(0, len(self.val) - self.context - 1,
                               (self.val_windows,), generator=generator)
        self.val_x, self.val_y = self._windows(self.val, starts)
        self.probe = self.val_x[_probe_indices(len(self.val_x))].numpy()

    def _windows(self, stream: torch.Tensor, starts: torch.Tensor):
        """Inputs and next-character targets for windows at the given starts."""
        idx = starts[:, None] + torch.arange(self.context)
        return stream[idx], stream[idx + 1]

    def batches(self, generator: torch.Generator):
        """One epoch of random training windows."""
        high = len(self.train) - self.context - 1
        for _ in range(self.steps):
            starts = torch.randint(0, high, (self.batch_size,), generator=generator)
            yield self._windows(self.train, starts)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))

    @staticmethod
    def hits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
        """Correct predictions and number of predictions."""
        return int((logits.argmax(-1) == targets).sum()), targets.numel()

    def evaluate(self, model) -> tuple[float, float]:
        """Validation accuracy and mean loss."""
        return _evaluate(model, self, self.val_x, self.val_y)


def _evaluate(model, task, xs: torch.Tensor, ys: torch.Tensor) -> tuple[float, float]:
    """Accuracy and mean loss over xs in eval mode, then back to train mode."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(xs), EVAL_CHUNK):
            x, y = xs[start:start + EVAL_CHUNK], ys[start:start + EVAL_CHUNK]
            logits = model(x)
            hit, count = task.hits(logits, y)
            correct += hit
            total += count
            loss_sum += float(task.loss(logits, y)) * count
    model.train()
    return correct / total, loss_sum / total


def _normalize(train: torch.Tensor, val: torch.Tensor, dims: tuple):
    """Scale uint8 images to float and standardize with training statistics."""
    train = train.float() / 255.0
    val = val.float() / 255.0
    mean = train.mean(dim=dims, keepdim=True)
    std = train.std(dim=dims, keepdim=True)
    return (train - mean) / std, (val - mean) / std


def _subset(x: torch.Tensor, y: torch.Tensor, size: int):
    """The same random examples every time, chosen once with seed 2."""
    generator = torch.Generator().manual_seed(2)
    idx = torch.randperm(len(x), generator=generator)[:size]
    return x[idx], y[idx]


def load_task(dataset: str, batch_size: int, context: int = 64, steps: int = 150,
              val_size: int | None = None):
    """Build the task for a dataset name: gtsrb, fashion, or shakespeare.

    val_size, if given, scores image tasks on a fixed random subset
    of the validation set to save time.
    """
    if dataset == "gtsrb":
        raw = _cached("gtsrb32", _build_gtsrb)
        train_x, val_x = _normalize(raw["train_x"], raw["test_x"], (0, 2, 3))
        val_y = raw["test_y"]
        if val_size is not None:
            val_x, val_y = _subset(val_x, val_y, val_size)
        return ImageTask(train_x, raw["train_y"], val_x, val_y, 43, batch_size)
    if dataset == "fashion":
        raw = _cached("fashion", _build_fashion)
        train_x, val_x = _normalize(raw["train_x"].flatten(1), raw["test_x"].flatten(1), (0, 1))
        return ImageTask(train_x, raw["train_y"], val_x, raw["test_y"], 10, batch_size)
    if dataset == "shakespeare":
        raw = _cached("shakespeare", _build_shakespeare)
        return TextTask(raw["train"], raw["test"], raw["vocab"], context, batch_size, steps)
    raise KeyError(dataset)
