import time
import statistics
from typing import Callable, Any, List, TypeVar
from collections import namedtuple
from tqdm import tqdm

T = TypeVar('T')
TimingResult = namedtuple('TimingResult', ['mean', 'std', 'min', 'max', 'runs', 'loops'])

def timeit(func, number = 100, repeat = 1) -> TimingResult:
    times: List[float] = []
    
    for _ in range(repeat):
        start_time = time.perf_counter()
        for _ in tqdm(range(number), desc = f'Repeat {_ + 1}'):
            func()
        end_time = time.perf_counter()
        
        iteration_time = ((end_time - start_time) / number) * 1_000_000
        times.append(iteration_time)
    
    result = TimingResult(
        mean=statistics.mean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0,
        min=min(times),
        max=max(times),
        runs=repeat,
        loops=number
    )
    print_timing_stats(result)
    return result

def print_timing_stats(result):
    time_unit = 'µs'
    values = [result.mean, result.std, result.min, result.max]
    
    if all(v > 1000 for v in values):
        values = [v / 1000 for v in values]
        time_unit = 'ms'
    
    print(f"{result.loops} loops, {result.runs} runs")
    print(f"Mean ± std: {values[0]:.3f} ± {values[1]:.3f} {time_unit}")
    print(f"Range: [{values[2]:.3f}, {values[3]:.3f}] {time_unit}")