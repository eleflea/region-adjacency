import torch
import numpy as np


_Tensor = torch.Tensor
_ndarray = np.ndarray


def check(r_np: _ndarray, r_torch: _ndarray):
    print(np.allclose(r_np, r_torch))
    diff = np.abs(r_np - r_torch)
    rel_diff = diff / np.maximum(np.abs(r_np), 1e-8)
    print(f'max abs error: {diff.max()}; rel abs error: {rel_diff.max()}')
    mask = rel_diff > 10.
    if len(mask) > 0:
        print(f'numpy: {r_np[mask][:10].tolist()}')
        print(f'cuda: {r_torch[mask][:10].tolist()}')

def mm_np(mask: _ndarray, features: _ndarray):
    f_square = np.sum(features ** 2, axis=-1)
    dis = f_square[:, None, ...] + f_square[..., None] - 2 * features @ np.transpose(features, (0, 2, 1))
    return mask * np.exp(-dis / 25)

def mm_torch(mask: _Tensor, features: _Tensor):
    f_square = features.pow(2).sum(-1)
    dis = f_square.unsqueeze(1) + f_square.unsqueeze(-1) - 2 * features.bmm(features.transpose(-2, -1))
    return mask * (-dis / 25).exp()

if __name__ == '__main__':
    features = np.random.randn(16, 256, 128).astype(np.float32)
    mask = np.random.randint(0, 2, size=(16, 256, 256)).astype(np.float32)
    features_cuda = torch.from_numpy(features).cuda()
    mask_cuda = torch.from_numpy(mask).cuda()
    r_np = mm_np(mask, features)
    r_torch = mm_torch(mask_cuda, features_cuda).cpu().numpy()
    check(r_np, r_torch)
