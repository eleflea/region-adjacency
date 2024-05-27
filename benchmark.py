from dataclasses import dataclass
from typing import Callable, Optional, Sequence, List

import numpy as np
import torch

from adj_matrix import (
    adj_matrix_numpy,
    adj_matrix_numpy_loop,
    adj_matrix_torch,
    adj_matrix_torch_cpp,
)
from profiler import Profiler


_Tensor = torch.Tensor
_ndarray = np.ndarray


@dataclass
class Case:
    """Class for benchmarking."""

    func: Callable
    args: Sequence
    times: int
    warmup: int
    name: Optional[str] = None
    is_cuda: bool = False
    primary: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = self.func.__name__


def to_numpy(t) -> _ndarray:
    if isinstance(t, _ndarray):
        return t
    if isinstance(t, _Tensor):
        return t.cpu().numpy()
    raise TypeError(f"`{type(t)}` is neither `numpy.ndarray` nor `torch.Tensor`")


def to_torch(t, cuda=False) -> _ndarray:
    t = torch.from_numpy(t)
    if cuda:
        return t.cuda()
    return t


def compare_results(expect, output, detail=True):
    expect = to_numpy(expect)
    output = to_numpy(output)
    all_close = np.allclose(expect, output)
    if not all_close and detail:
        abs_diff = np.abs(expect - output)
        rel_diff = abs_diff / np.maximum(np.abs(expect), 1e-8)
        print(f"max abs/rel error = {abs_diff.max()}/{rel_diff.max()}")
        mask = rel_diff > 0.5
        print("first 10 elements rel error > 0.5:")
        print(f"expect: {expect[mask][:10].tolist()}")
        print(f"output: {output[mask][:10].tolist()}")
    return all_close


def generate_inputs(segment_size, feature_size):
    segments = np.random.randint(feature_size[1], size=segment_size, dtype=np.int64)
    features = np.random.randn(*feature_size).astype(np.float32)
    return segments, features


def run_cases(cases):
    profilers: List[Profiler] = []
    expect_output = None

    for case in cases:
        p = Profiler(case.func, case.name)
        print(f"running {p.name}...", end=" ", flush=True)
        p.execute(case.args, case.times, case.warmup, case.is_cuda)
        profilers.append(p)
        if case.primary and expect_output is None:
            expect_output = p.output
        print("done", flush=True)

    for p in profilers:
        stat = p.statistic()
        print(f"'{p.name}' execute statistics:")
        print(
            f"\tavg = {stat['avg']:.3f} ms, min/max/std = {stat['min']:.3f}/{stat['max']:.3f}/{stat['std']:.3f} ms"
        )
        acc_status = "PASS" if compare_results(expect_output, p.output) else "FAIL"
        print(f"\tacc = {acc_status}")


def test():
    b, h, w = 8, 224, 224
    n = 256
    c = 128
    sigma = 5
    connectivity = 2
    segment_size = (b, h, w)
    feature_size = (b, n, c)

    segments, features = generate_inputs(segment_size, feature_size)
    args_np = (segments, features, sigma, connectivity)
    args_torch = (to_torch(segments), to_torch(features), sigma, connectivity)
    args_torch_cuda = (
        to_torch(segments, cuda=True),
        to_torch(features, cuda=True),
        sigma,
        connectivity,
    )

    cases = [
        Case(
            func=adj_matrix_numpy_loop,
            name="numpy-loop(cpu)",
            args=args_np,
            times=5,
            warmup=2,
            primary=True,
        ),
        Case(
            func=adj_matrix_numpy, name="numpy(cpu)", args=args_np, times=100, warmup=25
        ),
        Case(
            func=adj_matrix_torch,
            name="torch(cpu)",
            args=args_torch,
            times=100,
            warmup=25,
        ),
        Case(
            func=adj_matrix_torch,
            name="torch(gpu)",
            args=args_torch_cuda,
            times=200,
            warmup=20,
            is_cuda=True,
        ),
        Case(
            func=adj_matrix_torch_cpp,
            name="torch-cpp(gpu)",
            args=args_torch_cuda,
            times=200,
            warmup=20,
            is_cuda=True,
        ),
    ]
    run_cases(cases)


if __name__ == "__main__":
    test()
