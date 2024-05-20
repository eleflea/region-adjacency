import numpy as np
import torch
from torch.nn import functional as F


_Tensor = torch.Tensor
_ndarray = np.ndarray


def adj_matrix_numpy(segments: _ndarray, features: _ndarray, sigma: float):
    """
    Calculate the adjacency matrix from a graph.
    This method uses a 2x2 window to get a suboptimal result for speed.

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
        for y in range(h - 1):
            for x in range(w - 1):
                sub = segments[i, y: y + 2, x: x + 2]

                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)

                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if adj[i, idx1, idx2] != 0:
                        continue

                    diss = np.exp(-np.sum(np.square(features[i, idx1] - features[i, idx2])) / sigma ** 2)
                    adj[i, idx1, idx2] = adj[i, idx2, idx1] = diss

    return adj


def adj_matrix_torch(segments: _Tensor, features: _Tensor, sigma: float):
    """
    Calculate the adjacency matrix from a graph.
    This method uses a 2x2 window to get a suboptimal result for speed.

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

    b, n, c = features.shape
    # maxpool op does not support int type
    if not torch.is_floating_point(segments):
        segments = segments.float()

    # get max and min class indexes using a 2x2 window
    max_indexes = F.max_pool2d(segments.unsqueeze(1), 2, stride=1).long().view(b, -1)
    min_indexes = -F.max_pool2d(-segments.unsqueeze(1), 2, stride=1).long().view(b, -1)

    batch_indexes = torch.arange(b, dtype=torch.int64).view(b, 1, 1)
    c_indexes = torch.arange(c, dtype=torch.int64)

    # interleave max and min indexes
    left_mix_indexes = torch.cat((max_indexes, min_indexes), dim=-1)
    right_mix_indexes = torch.cat((min_indexes, max_indexes), dim=-1)
    # calc the postions in 1-d adj matrix
    n_indexes = left_mix_indexes * n + right_mix_indexes

    left_mix_features = features[batch_indexes, left_mix_indexes.unsqueeze(-1), c_indexes]
    right_mix_features = features[batch_indexes, right_mix_indexes.unsqueeze(-1), c_indexes]

    adj = torch.zeros(b, n, n, dtype=features.dtype, device=segments.device)
    # calc the distances based on features
    mask = left_mix_indexes != right_mix_indexes
    distances = torch.exp(-(left_mix_features - right_mix_features).pow(2).sum(-1) / sigma ** 2)
    adj.view(b, -1)[batch_indexes, n_indexes] = mask * distances
    return adj


if __name__ == '__main__':
    seg = torch.as_tensor([
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3]
    ], dtype=torch.int32)
    features = torch.randn(1, 6, 5)
    adj_torch = adj_matrix_torch(seg.view(1, 4, 4), features, 5).numpy()
    adj_np = adj_matrix_numpy(seg.squeeze(0).numpy(), features.squeeze(0).numpy(), 5, 6)
    print(np.allclose(adj_np, adj_torch))
