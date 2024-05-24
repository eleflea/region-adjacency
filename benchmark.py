import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, List

import numpy as np
import torch

# from approx_adj_matrix import adj_matrix_numpy, adj_matrix_torch
from adj_matrix import adj_matrix_numpy, adj_matrix_torch
from profiler import Profiler


_Tensor = torch.Tensor
_ndarray = np.ndarray


@dataclass
class Case:
    '''Class for benchmarking.'''
    func: Callable
    args: Sequence
    times: int
    warmup: int
    name: Optional[str] = None
    is_cuda: bool = False
    primary: bool = False


def to_numpy(t) -> _ndarray:
    if isinstance(t, _ndarray):
        return t
    if isinstance(t, _Tensor):
        return t.cpu().numpy()
    raise TypeError(f'`{type(t)}` is neither `numpy.ndarray` nor `torch.Tensor`')


def to_torch_cuda(t) -> _ndarray:
    return torch.from_numpy(t).cuda()


def compare_results(expect, output, detail=True):
    expect = to_numpy(expect)
    output = to_numpy(output)
    all_close = np.allclose(expect, output)
    if not all_close and detail:
        abs_diff = np.abs(expect - output)
        rel_diff = abs_diff / np.maximum(np.abs(expect), 1e-8)
        print(f'max abs/rel error = {abs_diff.max()}/{rel_diff.max()}')
        mask = rel_diff > 0.5
        print('first 10 elements rel error > 0.5:')
        print(f'expect: {expect[mask][:10].tolist()}')
        print(f'output: {output[mask][:10].tolist()}')
    return all_close


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
    b, h, w = 8, 224, 224
    n = 256
    c = 128
    sigma = 5
    segment_size = (b, h, w)
    feature_size = (b, n, c)

    segments, features = generate_inputs(segment_size, feature_size)
    args_np = (segments, features, sigma)
    args_torch = (to_torch_cuda(segments), to_torch_cuda(features), sigma)

    cases = [
        Case(func=adj_matrix_numpy, args=args_np, times=100, warmup=25, primary=True),
        Case(func=adj_matrix_torch, args=args_torch, times=200, warmup=25, is_cuda=True),
    ]
    profilers: List[Profiler] = []
    expect_output = None
    for case in cases:
        p = Profiler(case.func, case.name)
        print(f'running {p.name}...', end=' ', flush=True)
        p.execute(case.args, case.times, case.warmup, case.is_cuda)
        profilers.append(p)
        if case.primary and expect_output is None:
            expect_output = p.output
        print('done', flush=True)
    
    for p in profilers:
        p.summary()
        acc_status = 'PASS' if compare_results(expect_output, p.output) else 'FAIL'
        print(f'\tacc = {acc_status}')


if __name__ == '__main__':
    test()
