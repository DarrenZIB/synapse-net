import atexit
import os
import shutil
import tempfile
import uuid
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch_em
import torch_em.self_training as self_training
from elf.io import open_file
from torchvision import transforms

from .supervised_training import get_2d_model, get_3d_model, get_supervised_loader, _determine_ndim


_MRC_EXTENSIONS = (".mrc", ".rec")


def weak_augmentations(p: float = 0.75) -> callable:
    """The weak augmentations used in the unsupervised data loader.

    Args:
        p: The probability for applying one of the augmentations.

    Returns:
        The transformation function applying the augmentation.
    """
    norm = torch_em.transform.raw.standardize
    aug = transforms.Compose([
        norm,
        transforms.RandomApply([torch_em.transform.raw.GaussianBlur()], p=p),
        transforms.RandomApply([torch_em.transform.raw.AdditiveGaussianNoise(
            scale=(0, 0.15), clip_kwargs=False)], p=p
        ),
    ])
    return torch_em.transform.raw.get_raw_transform(normalizer=norm, augmentation1=aug)


def drop_mask_channel(x):
    """Keep only the raw channel of a stacked (raw, mask) patch."""
    return x[:1]


class ComposedTransform:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        for f in self.funcs:
            x = f(x)
        return x


class ChannelSplitterSampler:
    """Split a stacked (raw, mask) patch and pass both channels to the wrapped sampler."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, x):
        raw, mask = x[0], x[1]
        return self.sampler(raw, mask)


class MaskFractionSampler:
    """Accept a patch if at least `min_fraction` of its voxels lie inside the mask.

    Used as the default patch sampler when sample masks are given.

    Args:
        min_fraction: The minimal fraction of foreground mask voxels for accepting a patch.
    """
    def __init__(self, min_fraction: float = 0.5):
        self.min_fraction = min_fraction

    def __call__(self, raw, mask):
        mask = np.asarray(mask)
        return float((mask > 0).sum()) / mask.size >= self.min_fraction


def _load_volume(path: str, key: Optional[str]) -> np.ndarray:
    """Load a full volume. The key defaults to 'data' for mrc / rec files."""
    if key is None:
        if os.path.splitext(path)[1].lower() in _MRC_EXTENSIONS:
            key = "data"
        else:
            raise ValueError(f"Cannot derive the internal dataset key for {path}, please pass it explicitly.")
    with open_file(path, mode="r") as f:
        return f[key][:]


def _stack_raw_and_masks(data_paths, mask_paths, raw_key, mask_key, stack_dir):
    """Stack each tomogram with its mask as (2, Z, Y, X) and save it as hdf5 (dataset 'raw')."""
    if stack_dir is None:
        # tempfile respects $TMPDIR, so this can be redirected to node-local scratch on a cluster.
        stack_dir = tempfile.mkdtemp(prefix="synapse_net_masked_")
        atexit.register(shutil.rmtree, stack_dir, ignore_errors=True)
    else:
        os.makedirs(stack_dir, exist_ok=True)

    stacked_paths = []
    for i, (data_path, mask_path) in enumerate(zip(data_paths, mask_paths)):
        raw = _load_volume(data_path, raw_key)
        mask = _load_volume(mask_path, mask_key)
        if raw.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch between tomogram {data_path} {raw.shape} and mask {mask_path} {mask.shape}."
            )
        stacked = np.stack([raw, (mask > 0).astype(raw.dtype)], axis=0)

        out_path = os.path.join(stack_dir, f"stacked{i}_{uuid.uuid4().hex}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=stacked, chunks=True, compression="lzf")
        stacked_paths.append(out_path)

    return stacked_paths


def get_unsupervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    sample_mask_paths: Optional[Tuple[str]] = None,
    sampler: Optional[callable] = None,
    exclude_top_and_bottom: bool = False,
    mask_key: Optional[str] = None,
    stack_dir: Optional[str] = None,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for unsupervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        sample_mask_paths: The filepaths to the corresponding sample masks for each tomogram.
            Must be in the same order as `data_paths` and have the same shape as the tomograms.
            Nonzero voxels are treated as inside the mask.
        sampler: Accept or reject patches based on a condition. If masks are given, it is called as
            `sampler(raw, mask)`; if it is None, a `MaskFractionSampler()` is used.
            Without masks, it is called as `sampler(raw)`.
        exclude_top_and_bottom: Whether to exluce the five top and bottom slices to
            avoid artifacts at the border of tomograms.
        mask_key: The key that holds the mask inside of the mask files.
            Defaults to 'data' for mrc / rec files and to `raw_key` otherwise.
        stack_dir: Folder for the temporary stacked (raw, mask) files. By default a temporary
            folder is created (under $TMPDIR) and removed at exit.

    Returns:
        The PyTorch dataloader.
    """
    # We exclude the top and bottom slices where the tomogram reconstruction is bad.
    # TODO this seems unneccesary if we have a boundary mask - remove?
    if exclude_top_and_bottom:
        roi = np.s_[5:-5, :, :]
    else:
        roi = None

    if sample_mask_paths is not None:
        # Stack tomograms and masks and write them to temp files to use as input to RawDataset().
        assert len(data_paths) == len(sample_mask_paths), \
            f"Expected equal number of data_paths and and sample_masks_paths, got {len(data_paths)} data paths and {len(sample_mask_paths)} mask paths."  # noqa
        if mask_key is None and os.path.splitext(sample_mask_paths[0])[1].lower() not in _MRC_EXTENSIONS:
            mask_key = raw_key

        data_paths = tuple(_stack_raw_and_masks(data_paths, sample_mask_paths, raw_key, mask_key, stack_dir))
        raw_key = "raw"

        # Drop the mask channel BEFORE normalizing, so that the mask does not affect the raw statistics.
        # (The sampler runs on the un-transformed patch, so it still sees the mask.)
        base_transform = torch_em.transform.get_raw_transform()
        raw_transform = ComposedTransform(drop_mask_channel, base_transform)
        sampler = ChannelSplitterSampler(MaskFractionSampler() if sampler is None else sampler)
        with_channels = True
    else:
        raw_transform = torch_em.transform.get_raw_transform()
        with_channels = False

    _, ndim = _determine_ndim(patch_shape)
    transform = torch_em.transform.get_augmentations(ndim=ndim)

    if n_samples is None:
        n_samples_per_ds = None
    else:
        n_samples_per_ds = int(n_samples / len(data_paths))

    augmentations = (weak_augmentations(), weak_augmentations())

    datasets = [
        torch_em.data.RawDataset(path, raw_key, patch_shape, raw_transform, transform, roi=roi,
                                 n_samples=n_samples_per_ds, sampler=sampler, ndim=ndim,
                                 with_channels=with_channels, augmentations=augmentations)
        for path in data_paths
    ]
    ds = torch.utils.data.ConcatDataset(datasets)

    num_workers = 4 * batch_size
    loader = torch_em.segmentation.get_data_loader(ds, batch_size=batch_size,
                                                   num_workers=num_workers, shuffle=True)
    return loader


# TODO: use different paths for supervised and unsupervised training
# (We are currently not using this functionality directly, so this is not a high priority)
def semisupervised_training(
    name: str,
    train_paths: Tuple[str],
    val_paths: Tuple[str],
    label_key: str,
    patch_shape: Tuple[int, int, int],
    save_root: str,
    raw_key: str = "raw",
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    check: bool = False,
) -> None:
    """Run semi-supervised segmentation training.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the df5 files for the validation data.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the raw data inside of the hdf5.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        check: Whether to check the training and validation loaders instead of running training.
    """
    train_loader = get_supervised_loader(train_paths, raw_key, label_key, patch_shape, batch_size,
                                         n_samples=n_samples_train)
    val_loader = get_supervised_loader(val_paths, raw_key, label_key, patch_shape, batch_size,
                                       n_samples=n_samples_val)

    unsupervised_train_loader = get_unsupervised_loader(train_paths, raw_key, patch_shape, batch_size,
                                                        n_samples=n_samples_train)
    unsupervised_val_loader = get_unsupervised_loader(val_paths, raw_key, patch_shape, batch_size,
                                                      n_samples=n_samples_val)

    # TODO check the semisup loader
    if check:
        # from torch_em.util.debug import check_loader
        # check_loader(train_loader, n_samples=4)
        # check_loader(val_loader, n_samples=4)
        return

    # Check for 2D or 3D training
    is_2d = False
    z, y, x = patch_shape
    is_2d = z == 1

    if is_2d:
        model = get_2d_model(out_channels=2)
    else:
        model = get_3d_model(out_channels=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Self training functionality.
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=0.9)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        device=device,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
    )
    trainer.fit(n_iterations)