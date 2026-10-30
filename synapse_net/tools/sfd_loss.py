import math
import torch.nn.functional as F
import torch
import torch.nn as nn

_sfd_iters = 3
_sfd_smooth = 3
_separate_pool = True
_sfd_density_threshold = 0.5
gaussian_kernel_dict = {}
def _soft_erode(img, separate_pool=False):
    if separate_pool:
        p1 = -F.max_pool3d(
            -img, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)
        )
        p2 = -F.max_pool3d(
            -img, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)
        )
        p3 = -F.max_pool3d(
            -img, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)
        )
        return torch.min(torch.min(p1, p2), p3)
    p4 = -F.max_pool3d(-img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    return p4


def _soft_dilate(img):
    """Dilation via isotropic 3x3x3 max-pooling."""
    return F.max_pool3d(
        img, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
    )


def _soft_open(img, separate_pool=False):
    """Opening = dilate(erode(img))."""
    return _soft_dilate(_soft_erode(img, separate_pool=separate_pool))


def _skel(x, iters, separate_pool=False):
    img1 = _soft_open(x, separate_pool=separate_pool)
    skel = torch.relu(x - img1)
    for _ in range(iters):
        x = _soft_erode(x, separate_pool=separate_pool)
        img1 = _soft_open(x, separate_pool=separate_pool)
        delta = torch.relu(x - img1)
        skel = skel + torch.relu(delta - skel * delta)
    # save skel as mrc
    return skel


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    grid = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y, z = torch.meshgrid(grid, grid, grid)
    xyz_grid = torch.stack([x, y, z], dim=-1)
    g = torch.exp(-torch.sum(xyz_grid ** 2, dim=-1) / (2 * sigma ** 2))
    g /= (2 * math.pi * sigma ** 2) ** (3 / 2)
    g = g / torch.sum(g)
    return g


def _apply_gaussian_filter(
        seg: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    if (kernel_size, sigma) not in gaussian_kernel_dict:
        gaussian_kernel_dict[(kernel_size, sigma)] = _gaussian_kernel(
            kernel_size, sigma
        ).to(seg.device)
    g_kernel = gaussian_kernel_dict[(kernel_size, sigma)]
    g_kernel = g_kernel.view(1, 1, *g_kernel.size())
    padding = kernel_size // 2
    return F.conv3d(seg, g_kernel, padding=padding, groups=seg.shape[1])

def _surface_dice_loss( pred, target, separate_pool=False):


    out_channels = 1

    target_oh = target
    pred_prob = pred
    target_oh = torch.clamp(
            _apply_gaussian_filter(
                (target_oh == 1) * 1.0, kernel_size=15, sigma=2.0
            )
            * 1.5, max=1.0
    )
    target_oh = target_oh.detach()
    Sp = _skel(pred_prob, _sfd_iters, separate_pool=separate_pool)
    Sl = _skel(target_oh, _sfd_iters, separate_pool=separate_pool)
    Tprec = (torch.sum(Sp * target_oh) + _sfd_smooth) / (torch.sum(Sp) + _sfd_smooth)
    Trec = (torch.sum(Sl * pred_prob) + _sfd_smooth) / (torch.sum(Sl) + _sfd_smooth)
    dice = 2.0 * Tprec * Trec / (Tprec + Trec)
    thr = _sfd_density_threshold
    total_voxels = float(Sp.numel())
    sp_density = (Sp > thr).sum().item() / (total_voxels + 1e-6)
    sl_density = (Sl > thr).sum().item() / (total_voxels + 1e-6)
    return 1.0 - dice


class SurfaceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        return _surface_dice_loss(
            pred[:, 1:2],
            target[:, 1:2],
            separate_pool=_separate_pool,
        )

