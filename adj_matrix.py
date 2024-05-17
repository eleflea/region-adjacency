import numpy as np
import torch
from torch.nn import functional as F

def get_A(segments, features, sigma, node_count):
    '''
        根据 segments 判定邻接矩阵
    :return:
    '''
    A = np.zeros([node_count, node_count], dtype=np.float32)
    (h, w) = segments.shape

    for i in range(h - 1):
        for j in range(w - 1):
            sub = segments[i:i + 2, j:j + 2]
            sub = sub

            sub_max = np.max(sub).astype(np.int32)
            sub_min = np.min(sub).astype(np.int32)

            # if len(sub_set)>1:
            if sub_max != sub_min:
                idx1 = sub_max
                idx2 = sub_min
                if A[idx1, idx2] != 0:
                    continue

                pix1 = features[idx1]
                pix2 = features[idx2]
                diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                A[idx1, idx2] = A[idx2, idx1] = diss

    return A

def get_adj_matrix(segments, features, sigma):
    """
    Get adjacency matrix from graph.

    Args:
        segments (Tensor[B, H, W]): A RAG (Region Adjacency Graph).
        features (Tensor[B, N, C]): A tensor.
        sigma (float): sigma.
    Returns:
        A Tensor(B, N, N) represents adjacency matrix.
    """
    b, n, c = features.shape
    if not torch.is_floating_point(segments):
        segments = segments.float()
    max_indexes = F.max_pool2d(segments.unsqueeze(1), 2, stride=1).long().view(b, -1)
    min_indexes = -F.max_pool2d(-segments.unsqueeze(1), 2, stride=1).long().view(b, -1)
    adj = torch.zeros(b, n, n, dtype=features.dtype, device=segments.device)
    batch_indexes = torch.arange(b, dtype=torch.int64).view(b, 1, 1)
    c_indexes = torch.arange(c, dtype=torch.int64)
    left_mix_indexes = torch.cat((max_indexes, min_indexes), dim=-1)
    right_mix_indexes = torch.cat((min_indexes, max_indexes), dim=-1)
    n_indexes = left_mix_indexes * n + right_mix_indexes
    left_mix_features = features[batch_indexes, left_mix_indexes.unsqueeze(-1), c_indexes]
    right_mix_features = features[batch_indexes, right_mix_indexes.unsqueeze(-1), c_indexes]
    adj.view(b, -1)[batch_indexes, n_indexes] = (left_mix_indexes != right_mix_indexes) * torch.exp(-(left_mix_features - right_mix_features).pow(2).sum(-1) / sigma ** 2)
    return adj


if __name__ == '__main__':
    seg = torch.as_tensor([
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3]
    ], dtype=torch.int32)
    features = torch.randn(1, 6, 5)
    adj_torch = get_adj_matrix(seg.view(1, 4, 4), features, 5).numpy()
    adj_np = get_A(seg.squeeze(0).numpy(), features.squeeze(0).numpy(), 5, 6)
    print(np.allclose(adj_np, adj_torch))