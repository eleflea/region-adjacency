from dataclasses import dataclass
from typing import Callable, Optional, Sequence, List

import numpy as np
import torch

from region_adjacency import (
    region_adjacency_numpy,
    region_adjacency_numpy_loop,
    region_adjacency_torch,
    region_adjacency_torch_cpp,
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


def generate_inputs(segment_size, num_labels):
    segments = np.random.randint(num_labels, size=segment_size, dtype=np.int64)
    return segments


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
        acc_status = "PASS" if expect_output is None or compare_results(expect_output, p.output) else "FAIL"
        print(f"\tacc = {acc_status}")


def test():
    b, h, w = 8, 224, 224
    num_labels = 256
    connectivity = 2
    segment_size = (b, h, w)

    segments = generate_inputs(segment_size, num_labels)
    args_np = (segments, num_labels, connectivity)
    args_torch = (to_torch(segments), num_labels, connectivity)
    args_torch_cuda = (
        to_torch(segments, cuda=True),
        num_labels,
        connectivity,
    )

    cases = [
        Case(
            func=region_adjacency_numpy_loop,
            name="numpy-loop(cpu)",
            args=args_np,
            times=5,
            warmup=2,
            primary=True,
        ),
        Case(
            func=region_adjacency_numpy, name="numpy(cpu)", args=args_np, times=100, warmup=25
        ),
        Case(
            func=region_adjacency_torch,
            name="torch(cpu)",
            args=args_torch,
            times=100,
            warmup=25,
        ),
        Case(
            func=region_adjacency_torch,
            name="torch(gpu)",
            args=args_torch_cuda,
            times=200,
            warmup=20,
            is_cuda=True,
        ),
        Case(
            func=region_adjacency_torch_cpp,
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
