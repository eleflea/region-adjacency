from typing import Callable, Optional, Sequence, Any, Dict
import time

import torch
import numpy as np


class Profiler:

    def __init__(self, func: Callable, name: Optional[str] = None) -> None:
        self.func = func
        self.name = func.__name__ if name is None else name
        self.output = None
        self.records = []

    def execute(
        self, args: Sequence[Any], times: int, warmup: int = 0, is_cuda: bool = False
    ):
        if warmup == 0:
            self.output = self.func(*args)
        else:
            for _ in range(warmup):
                self.output = self.func(*args)

        if is_cuda:
            torch.cuda.synchronize()

        for _ in range(times):
            start = time.time()
            self.func(*args)
            if is_cuda:
                torch.cuda.synchronize()
            self.records.append(time.time() - start)

        return self

    def statistic(self) -> Dict:
        times = len(self.records)
        stat = {}
        stat["times"] = times
        if times == 0:
            return stat

        stat["avg"] = np.mean(self.records).item() * 100
        stat["min"] = np.min(self.records).item() * 100
        stat["max"] = np.max(self.records).item() * 100
        stat["std"] = np.std(self.records).item() * 100
        stat["sum"] = np.sum(self.records).item() * 100
        return stat

    def summary(self) -> None:
        stat = self.statistic()
        print(
            f"'{self.name}' execute statistics:\n\tavg = {stat['avg']:.3f} ms, min/max/std = {stat['min']:.3f}/{stat['max']:.3f}/{stat['std']:.3f} ms"
        )
