import numpy as np
import torch
import time

from adj_matrix import get_A, get_adj_matrix

def generate_inputs(segment_size, feature_size):
    segments = np.random.randint(feature_size[1], size=segment_size, dtype=np.int32)
    features = np.random.randn(*feature_size).astype(np.float32)
    return segments, features

def timeit(fn, args, times=100):
    for _ in range(20):
        fn(*args)
    start = time.time()
    for _ in range(times):
        fn(*args)
    return (time.time() - start) / times, fn(*args)

def test():
    n = 224
    sigma = 5
    segment_size = (1, 256, 256)
    feature_size = (1, n, 128)
    segments, features = generate_inputs(segment_size, feature_size)
    t_np, r_np = timeit(get_A, (segments[0], features[0], sigma, n))
    time.sleep(2)
    segments_torch = torch.from_numpy(segments)
    features_torch = torch.from_numpy(features)
    t_torch, r_torch = timeit(get_adj_matrix, (segments_torch, features_torch, sigma))
    time.sleep(2)
    segments_torch = segments_torch.cuda()
    features_torch = features_torch.cuda()
    t_torch_cuda, r_torch_cuda = timeit(get_adj_matrix, (segments_torch, features_torch, sigma))
    print(f'numpy time: {t_np: .6f}s\ntorch(cpu) time: {t_torch: .6f}s\ntorch(cuda) time: {t_torch_cuda: .6f}s\nspeedup={t_np / t_torch_cuda: .3f}.')
    print(np.allclose(r_np, r_torch.numpy()))
    print(np.allclose(r_np, r_torch_cuda.cpu().numpy()))

if __name__ == '__main__':
    test()
