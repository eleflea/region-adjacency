import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load


_adj_matrix_cpp = load(name='_adj_matrix_cpp', sources=['adj_matrix_cuda.cpp', 'adj_matrix_cuda_kernel.cu'])

_Tensor = torch.Tensor
_ndarray = np.ndarray


def adj_matrix_numpy(segments: _ndarray, features: _ndarray, sigma: float, connectivity: int = 1):
    """
    Calculate the adjacency matrix from a graph.

    Args:
        segments (ndarray[B, H, W]):
            A RAG (Region Adjacency Graph). Where B is batch size; H and W represent height and width of the graphs.
            The label of the graphs should be a class index in the range [0, N).
        features (ndarray[B, N, C]):
            A ndarray represents features for each node.
            Where B is batch size; N is number of nodes; C is dimension of the feature.
        sigma (float): A parameter used to normalize the distance.
        connectivity (int): The connectivity between pixels in segments.
            For a 2D image, a connectivity of 1 corresponds to immediate neighbors up, down, left, and right,
            while a connectivity of 2 also includes diagonal neighbors.
    Returns:
        A ndarray(B, N, N) represents the adjacency matrix.
    """

    if connectivity not in {1, 2}:
        raise RuntimeError(f'Unexpected connectivity {connectivity}, should be 1 or 2.')

    _4_connect = np.array([(-1, 0), (0, -1), (1, 0), (0, 1)], dtype=np.int64)
    _8_connect = np.array([(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)], dtype=np.int64)
    connect = _4_connect if connectivity == 1 else _8_connect

    b, h, w = segments.shape
    n = features.shape[1]
    adj = np.zeros([b, n, n], dtype=np.float32)

    h_i = np.arange(h, dtype=np.int64)
    w_i = np.arange(w, dtype=np.int64)
    hw_i = np.stack(np.meshgrid(w_i, h_i, indexing='xy')[::-1], axis=-1)

    neighbor_i = hw_i[..., None, :] + connect # (H, W, 8, 2)
    ny_i, nx_i = neighbor_i[..., 0], neighbor_i[..., 1]
    np.clip(ny_i, 0, h - 1, out=ny_i)
    np.clip(nx_i, 0, w - 1, out=nx_i)
    b_i = np.arange(b, dtype=np.int64).reshape(b, 1, 1, 1)
    neighbor_vals = segments[b_i, ny_i, nx_i] # (B, H, W, 8)
    vals = segments[..., None]
    is_diff = neighbor_vals != vals
    adj[b_i, neighbor_vals, vals] = is_diff
    adj[b_i, vals, neighbor_vals] = is_diff

    f_square = np.sum(features ** 2, axis=-1)
    dis = f_square[:, None, ...] + f_square[..., None] - 2 * features @ np.transpose(features, (0, 2, 1))
    adj *= np.exp(-dis / sigma ** 2)

    return adj


def adj_matrix_torch(segments: _Tensor, features: _Tensor, sigma: float):
    """
    Calculate the adjacency matrix from a graph.

    Args:
        segments (Tensor[B, H, W]):
            A RAG (Region Adjacency Graph). Where B is batch size; H and W represent height and width of the graphs.
            The label of the graphs should be a class index in the range [0, N).
        features (Tensor[B, N, C]):
            A tensor represents features for each node.
            Where B is batch size; N is number of nodes; C is dimension of the feature.
        sigma (float): A parameter used to normalize the distance.
    Returns:
        A Tensor(B, N, N) represents the adjacency matrix.
    """

    return _adj_matrix_cpp.forward(segments, features, sigma)
