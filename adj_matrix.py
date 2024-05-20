import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load


_adj_matrix_cpp = load(name='_adj_matrix_cpp', sources=['adj_matrix_cuda.cpp', 'adj_matrix_cuda_kernel.cu'])

_Tensor = torch.Tensor
_ndarray = np.ndarray


def adj_matrix_numpy(segments: _ndarray, features: _ndarray, sigma: float):
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
    Returns:
        A ndarray(B, N, N) represents the adjacency matrix.
    """

    b, h, w = segments.shape
    n = features.shape[1]
    adj = np.zeros([b, n, n], dtype=np.float32)

    for i in range(b):
        for y in range(h):
            for x in range(w):
                v = segments[i, y, x]
                if y - 1 >= 0 and segments[i, y - 1, x] != v:
                    nv = segments[i, y - 1, x]
                    adj[i, nv, v] = 1
                    adj[i, v, nv] = 1
                if y + 1 < h and segments[i, y + 1, x] != v:
                    nv = segments[i, y + 1, x]
                    adj[i, nv, v] = 1
                    adj[i, v, nv] = 1
                if x - 1 >= 0 and segments[i, y, x - 1] != v:
                    nv = segments[i, y, x - 1]
                    adj[i, nv, v] = 1
                    adj[i, v, nv] = 1
                if x + 1 < w and segments[i, y, x + 1] != v:
                    nv = segments[i, y, x + 1]
                    adj[i, nv, v] = 1
                    adj[i, v, nv] = 1

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
