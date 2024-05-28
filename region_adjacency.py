from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load


_region_adjacency_cpp = load(
    name="_region_adjacency_cpp",
    sources=["region_adjacency_cuda.cpp", "region_adjacency_cuda_kernel.cu"],
)

_Tensor = torch.Tensor
_ndarray = np.ndarray


def region_adjacency_numpy_loop(
    labelled_imgs: _ndarray, num_labels: Optional[int] = None, connectivity: int = 1
) -> _ndarray:
    """
    Calculate the region adjacency matrices from labelled images.

    Args:
        labelled_imgs (ndarray[B, H, W]):
            labelled images, where each pixel is assigned the integer label of the region it belongs to.
            Where B is batch size; H and W represent height and width of the images.
            The labels of the images should be a integer class index in the range [0, N).
        num_labels (int, optional): The number of labels, which is equals to N.
            If it is `None`, the number of labels will be the maximum of `labelled_imgs`. Default: None.
        connectivity (int, optional): The connectivity between pixels in labelled images.
            For a 2D image, a connectivity of 1 corresponds to immediate neighbors up, down, left, and right,
            while a connectivity of 2 also includes diagonal neighbors. Default: 1.
    Returns:
        A ndarray(B, N, N) represents the region adjacency matrices.
    """

    if connectivity not in {1, 2}:
        raise RuntimeError(f"Unexpected connectivity {connectivity}, should be 1 or 2.")

    _4_connect = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    _8_connect = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
    connect = _4_connect if connectivity == 1 else _8_connect

    b, h, w = labelled_imgs.shape
    n = np.max(labelled_imgs).astype(np.int64) + 1 if num_labels is None else num_labels
    adj = np.zeros([b, n, n], dtype=labelled_imgs.dtype)

    for i in range(b):
        for y in range(h):
            for x in range(w):
                for d_y, d_x in connect:
                    other_y = d_y + y
                    other_x = d_x + x
                    if not (0 <= other_y < h and 0 <= other_x < w):
                        continue
                    other_v = labelled_imgs[i, other_y, other_x]
                    v = labelled_imgs[i, y, x]
                    if other_v == v:
                        continue

                    adj[i, v, other_v] = adj[i, other_v, v] = 1

    return adj


def region_adjacency_numpy(
    labelled_imgs: _ndarray, num_labels: Optional[int] = None, connectivity: int = 1
) -> _ndarray:
    """
    Calculate the region adjacency matrices from labelled images.

    Args:
        labelled_imgs (ndarray[B, H, W]):
            labelled images, where each pixel is assigned the integer label of the region it belongs to.
            Where B is batch size; H and W represent height and width of the images.
            The labels of the images should be a integer class index in the range [0, N).
        num_labels (int, optional): The number of labels, which is equals to N.
            If it is `None`, the number of labels will be the maximum of `labelled_imgs`. Default: None.
        connectivity (int, optional): The connectivity between pixels in labelled images.
            For a 2D image, a connectivity of 1 corresponds to immediate neighbors up, down, left, and right,
            while a connectivity of 2 also includes diagonal neighbors. Default: 1.
    Returns:
        A ndarray(B, N, N) represents the region adjacency matrices.
    """

    if connectivity not in {1, 2}:
        raise RuntimeError(f"Unexpected connectivity {connectivity}, should be 1 or 2.")

    _4_connect = np.array([(-1, 0), (0, -1), (1, 0), (0, 1)], dtype=np.int64)
    _8_connect = np.array(
        [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)],
        dtype=np.int64,
    )
    connect = _4_connect if connectivity == 1 else _8_connect

    b, h, w = labelled_imgs.shape
    n = np.max(labelled_imgs).astype(np.int64) + 1 if num_labels is None else num_labels
    adj = np.zeros([b, n, n], dtype=labelled_imgs.dtype)

    h_i = np.arange(h, dtype=np.int64)
    w_i = np.arange(w, dtype=np.int64)
    hw_i = np.stack(np.meshgrid(w_i, h_i, indexing="ij"), axis=-1)

    neighbor_i = hw_i[..., None, :] + connect  # (H, W, 8, 2)
    ny_i, nx_i = neighbor_i[..., 0], neighbor_i[..., 1]
    np.clip(ny_i, 0, h - 1, out=ny_i)
    np.clip(nx_i, 0, w - 1, out=nx_i)
    b_i = np.arange(b, dtype=np.int64).reshape(b, 1, 1, 1)
    neighbor_vals = labelled_imgs[b_i, ny_i, nx_i]  # (B, H, W, 8)
    vals = labelled_imgs[..., None]
    is_diff = neighbor_vals != vals
    adj[b_i, neighbor_vals, vals] = is_diff
    adj[b_i, vals, neighbor_vals] = is_diff

    return adj


def region_adjacency_torch(
    labelled_imgs: _Tensor, num_labels: Optional[int] = None, connectivity: int = 1
) -> _Tensor:
    """
    Calculate the region adjacency matrices from labelled images.

    Args:
        labelled_imgs (Tensor[B, H, W]):
            labelled images, where each pixel is assigned the integer label of the region it belongs to.
            Where B is batch size; H and W represent height and width of the images.
            The labels of the images should be a integer class index in the range [0, N).
        num_labels (int, optional): The number of labels, which is equals to N.
            If it is `None`, the number of labels will be the maximum of `labelled_imgs`. Default: None.
        connectivity (int, optional): The connectivity between pixels in labelled images.
            For a 2D image, a connectivity of 1 corresponds to immediate neighbors up, down, left, and right,
            while a connectivity of 2 also includes diagonal neighbors. Default: 1.
    Returns:
        A Tensor(B, N, N) represents the region adjacency matrices.
    """

    if connectivity not in {1, 2}:
        raise RuntimeError(f"Unexpected connectivity {connectivity}, should be 1 or 2.")

    _4_connect = torch.as_tensor([(-1, 0), (0, -1), (1, 0), (0, 1)], dtype=torch.int64)
    _8_connect = torch.as_tensor(
        [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)],
        dtype=torch.int64,
    )
    connect = _4_connect if connectivity == 1 else _8_connect

    b, h, w = labelled_imgs.shape
    n = labelled_imgs.max().long() + 1 if num_labels is None else num_labels
    adj = torch.zeros(b, n, n, dtype=labelled_imgs.dtype, device=labelled_imgs.device)

    h_i = torch.arange(h, dtype=torch.int64)
    w_i = torch.arange(w, dtype=torch.int64)
    hw_i = torch.stack(torch.meshgrid(w_i, h_i, indexing="ij"), dim=-1)

    neighbor_i = hw_i.unsqueeze(-2) + connect  # (H, W, 8, 2)
    ny_i, nx_i = neighbor_i[..., 0], neighbor_i[..., 1]
    ny_i.clamp_(0, h - 1)
    nx_i.clamp_(0, w - 1)
    b_i = torch.arange(b, dtype=torch.int64).view(b, 1, 1, 1)
    neighbor_vals = labelled_imgs[b_i, ny_i, nx_i]  # (B, H, W, 8)
    vals = labelled_imgs.unsqueeze(-1)
    is_diff = (neighbor_vals != vals).to(labelled_imgs.dtype)
    adj[b_i, neighbor_vals, vals] = is_diff
    adj[b_i, vals, neighbor_vals] = is_diff

    return adj


def region_adjacency_torch_cpp(
    labelled_imgs: _Tensor, num_labels: Optional[int] = None, connectivity: int = 1
) -> _Tensor:
    """
    Calculate the region adjacency matrices from labelled images.

    Args:
        labelled_imgs (Tensor[B, H, W]):
            labelled images, where each pixel is assigned the integer label of the region it belongs to.
            Where B is batch size; H and W represent height and width of the images.
            The labels of the images should be a integer class index in the range [0, N).
        num_labels (int, optional): The number of labels, which is equals to N.
            If it is `None`, the number of labels will be the maximum of `labelled_imgs`. Default: None.
        connectivity (int, optional): The connectivity between pixels in labelled images.
            For a 2D image, a connectivity of 1 corresponds to immediate neighbors up, down, left, and right,
            while a connectivity of 2 also includes diagonal neighbors. Default: 1.
    Returns:
        A Tensor(B, N, N) represents the region adjacency matrices.
    """

    if num_labels is None:
        num_labels = 0

    return _region_adjacency_cpp.forward(labelled_imgs, num_labels, connectivity)
