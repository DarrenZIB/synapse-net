import numpy as np
import torch
import torch.nn as nn


class BottleneckHook:
    """Records the output of the last Conv3d in model.base for every forward pass."""

    def __init__(self, model: nn.Module, flatten: bool = True):
        conv = [m for m in model.base.modules() if isinstance(m, nn.Conv3d)][-1]
        self.flatten = flatten
        vectors = self.vectors = []

        # A plain function (not a bound method), so the hook still writes into this
        # list when torch_em runs a deep copy of the model.
        def hook(module, inputs, output):
            feats = output.detach().float().cpu()
            vec = feats.flatten(start_dim=1) if flatten else feats.mean(dim=(2, 3, 4))
            vectors.append(vec.numpy())

        self._handle = conv.register_forward_hook(hook)

    def remove(self):
        self._handle.remove()

    def result(self) -> np.ndarray:
        return np.concatenate(self.vectors, axis=0)