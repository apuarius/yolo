from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics import YOLO, settings
from ultralytics.data.augment import BaseTransform, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER, YAML, colorstr
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "yolo11s_seg_train.yaml"


class RandomRequestedGeometry(RandomPerspective):
    """Apply one segmentation-safe affine warp using a random subset of requested operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0)
        self.rotate90_p = _probability(config, "rotate90_probability")
        self.small_rotation_p = _probability(config, "small_rotation_probability")
        self.small_rotation_degrees = _non_negative(config, "small_rotation_degrees")
        self.scale_p = _probability(config, "scale_probability")
        self.scale_limit = _non_negative(config, "scale_limit")
        self.translate_p = _probability(config, "translate_probability")
        self.translate_limit = _non_negative(config, "translate_limit")

    def _compute_affine_matrix(self, img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -img.shape[1] / 2
        center[1, 2] = -img.shape[0] / 2

        if random.random() < self.rotate90_p:
            angle = random.choice((-90.0, 90.0))
        elif random.random() < self.small_rotation_p:
            angle = random.uniform(-self.small_rotation_degrees, self.small_rotation_degrees)
        else:
            angle = 0.0

        scale = (
            random.uniform(1.0 - self.scale_limit, 1.0 + self.scale_limit)
            if random.random() < self.scale_p
            else 1.0
        )
        rotation = np.eye(3, dtype=np.float32)
        rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)

        translation = np.eye(3, dtype=np.float32)
        if random.random() < self.translate_p:
            translation[0, 2] = random.uniform(
                0.5 - self.translate_limit, 0.5 + self.translate_limit
            ) * size[0]
            translation[1, 2] = random.uniform(
                0.5 - self.translate_limit, 0.5 + self.translate_limit
            ) * size[1]
        else:
            translation[0, 2] = size[0] / 2
            translation[1, 2] = size[1] / 2

        return translation @ rotation @ center, scale


class RandomRequestedColor(BaseTransform):
    """Apply the requested image-only color transforms with independent probabilities."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.brightness_p = _probability(config, "brightness_probability")
        self.brightness_limit = _non_negative(config, "brightness_limit")
        self.contrast_p = _probability(config, "contrast_probability")
        self.contrast_limit = _non_negative(config, "contrast_limit")
        self.hsv_p = _probability(config, "hsv_probability")
        self.hue_shift_limit = int(config["hue_shift_limit"])
        self.saturation_shift_limit = int(config["saturation_shift_limit"])
        self.value_shift_limit = int(config["value_shift_limit"])
        self.gamma_p = _probability(config, "gamma_probability")
        self.gamma_limit = tuple(float(value) / 100.0 for value in config["gamma_limit"])
        if len(self.gamma_limit) != 2 or min(self.gamma_limit) <= 0:
            raise ValueError(f"gamma_limit must contain two positive values, got {config['gamma_limit']}.")

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 3:
            return labels

        image_float = image.astype(np.float32)
        if random.random() < self.brightness_p:
            image_float += random.uniform(-self.brightness_limit, self.brightness_limit) * 255.0
        if random.random() < self.contrast_p:
            factor = random.uniform(1.0 - self.contrast_limit, 1.0 + self.contrast_limit)
            image_float = (image_float - 127.5) * factor + 127.5
        image = np.clip(image_float, 0, 255).astype(np.uint8)

        if random.random() < self.hsv_p:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (
                hsv[..., 0] + random.randint(-self.hue_shift_limit, self.hue_shift_limit)
            ) % 180
            hsv[..., 1] = np.clip(
                hsv[..., 1]
                + random.randint(-self.saturation_shift_limit, self.saturation_shift_limit),
                0,
                255,
            )
            hsv[..., 2] = np.clip(
                hsv[..., 2] + random.randint(-self.value_shift_limit, self.value_shift_limit),
                0,
                255,
            )
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if random.random() < self.gamma_p:
            gamma = random.uniform(*sorted(self.gamma_limit))
            lookup = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            image = cv2.LUT(image, lookup)

        labels["img"] = image
        return labels


class OnlineAugmentedSegmentationDataset(YOLODataset):
    """Repeat each source image online while preserving independent random transforms."""

    views_per_image = 4
    expected_epoch_samples: int | None = None
    geometry_config: dict[str, Any] = {}
    color_config: dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.augment:
            return

        epoch_samples = self.ni * self.views_per_image
        if self.expected_epoch_samples is not None and epoch_samples != self.expected_epoch_samples:
            raise ValueError(
                "Training sample count mismatch: "
                f"{self.ni} source images x {self.views_per_image} views = {epoch_samples}, "
                f"but expected_epoch_samples={self.expected_epoch_samples}."
            )
        LOGGER.info(
            f"train: online views {self.ni} source images x {self.views_per_image} = {epoch_samples} samples/epoch"
        )

    def __len__(self) -> int:
        source_count = super().__len__()
        return source_count * self.views_per_image if self.augment else source_count

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        return super().get_image_and_label(index % self.ni)

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            # Insert before color transforms and flips. RandomPerspective updates polygons and boxes together.
            transforms.insert(3, RandomRequestedGeometry(self.geometry_config))
            transforms.insert(4, RandomRequestedColor(self.color_config))
        return transforms


class OnlineSegmentationTrainer(SegmentationTrainer):
    """Use the repeated augmented dataset for train and the stock dataset for val."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        dataset_class = OnlineAugmentedSegmentationDataset if mode == "train" else YOLODataset
        return dataset_class(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


def _probability(config: dict[str, Any], key: str) -> float:
    value = float(config[key])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {value}.")
    return value


def _non_negative(config: dict[str, Any], key: str) -> float:
    value = float(config[key])
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative, got {value}.")
    return value


def load_training_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")

    config = YAML.load(path)
    augmentation = config.pop("online_augmentation")
    evaluation = config.pop("evaluation", {})

    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {data_path}")
    config["data"] = str(data_path)

    project_path = Path(config["project"])
    if not project_path.is_absolute():
        project_path = ROOT / project_path
    config["project"] = str(project_path)

    views = int(augmentation["views_per_image"])
    if views < 1:
        raise ValueError(f"views_per_image must be at least 1, got {views}.")
    OnlineAugmentedSegmentationDataset.views_per_image = views
    OnlineAugmentedSegmentationDataset.expected_epoch_samples = int(augmentation["expected_epoch_samples"])
    OnlineAugmentedSegmentationDataset.geometry_config = augmentation["geometry"]
    OnlineAugmentedSegmentationDataset.color_config = augmentation["color"]
    config["augmentations"] = []
    return config, evaluation


def check_tensorboard() -> None:
    try:
        import tensorboard  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required. Install it once with: pip install tensorboard"
        ) from exc
    settings.update({"tensorboard": True})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg with the project augmentation policy.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_tensorboard()
    train_args, evaluation = load_training_config(args.config.resolve())

    LOGGER.info(f"Training config: {args.config.resolve()}")
    LOGGER.info(f"TensorBoard log root: {train_args['project']}")
    LOGGER.info("Validation uses the stock non-augmented YOLODataset pipeline.")

    model = YOLO(train_args.pop("model"))
    model.train(trainer=OnlineSegmentationTrainer, **train_args)

    save_dir = Path(model.trainer.save_dir)
    LOGGER.info(f"Training complete. TensorBoard: tensorboard --logdir {save_dir.parent}")

    if evaluation.get("test_after_training", False):
        best_weights = save_dir / "weights" / "best.pt"
        YOLO(best_weights).val(
            data=train_args["data"],
            split="test",
            imgsz=train_args["imgsz"],
            batch=train_args["batch"],
            device=train_args["device"],
            project=str(save_dir.parent),
            name=f"{save_dir.name}_test",
            augment=False,
        )


if __name__ == "__main__":
    main()
