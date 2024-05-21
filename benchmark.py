import numpy as np
import torch
import time

# from approx_adj_matrix import adj_matrix_numpy, adj_matrix_torch
from adj_matrix import adj_matrix_numpy, adj_matrix_torch

def generate_inputs(segment_size, feature_size):
    segments = np.random.randint(feature_size[1], size=segment_size, dtype=np.int64)
    features = np.random.randn(*feature_size).astype(np.float32)
    return segments, features

def timeit(fn, args, times=100, cuda=False):
    for _ in range(20):
        fn(*args)
    if cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(times):
        fn(*args)
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - start) / times, fn(*args)

def test():
    n = 256
    sigma = 5
    segment_size = (8, 224, 224)
    feature_size = (8, n, 128)
    segments, features = generate_inputs(segment_size, feature_size)
    t_np, r_np = timeit(adj_matrix_numpy, (segments, features, sigma))
    time.sleep(2)
    segments_torch = torch.from_numpy(segments)
    features_torch = torch.from_numpy(features)
    segments_torch = segments_torch.cuda()
    features_torch = features_torch.cuda()
    t_torch_cuda, r_torch_cuda = timeit(adj_matrix_torch, (segments_torch, features_torch, sigma))
    print(f'numpy time: {t_np: .6f}s\ntorch(cuda) time: {t_torch_cuda: .6f}s\nspeedup={t_np / t_torch_cuda: .3f}.')
    r_torch_cuda = r_torch_cuda.cpu().numpy()
    print(np.allclose(r_np, r_torch_cuda))
    diff = np.abs(r_np - r_torch_cuda)
    rel_diff = diff / np.maximum(np.abs(r_np), 1e-8)
    print(f'max abs error: {diff.max()}; rel abs error: {rel_diff.max()}')
    mask = rel_diff > 10.
    print(f'numpy: {r_np[mask][:10].tolist()}')
    print(f'cuda: {r_torch_cuda[mask][:10].tolist()}')

if __name__ == '__main__':
    test()
