from typing import Any, List, Callable, Union
import math
import numpy as np
from numpy import ndarray
import torch
import random
import time

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def divide(items: ndarray, ratio: float, seed: int = 0) -> List[ndarray]:
    item_num = len(items)
    rdm = np.random.RandomState(seed)
    idx = np.arange(item_num)
    rdm.shuffle(idx)
    div_num = int(ratio * item_num)
    trn, evl = items[idx[: div_num]], items[idx[div_num:]]
    return trn, evl

class Averager:
    def __init__(self) -> None:
        self.__total = 0.0
        self.__num = 0
    
    def join(self, val: float, weight: float = 1.0) -> None:
        self.__total += (val * weight)
        self.__num += weight
    
    def get(self) -> float:
        return 0.0 if self.__num == 0 else self.__total / self.__num

class Maximizer:
    def __init__(self) -> None:
        self.__max = float('-inf')
    
    def join(self, *vals: List[float]) -> bool:
        val = max(vals)
        update_required = val > self.__max
        if update_required: self.__max = val
        return update_required
        
    def get(self) -> float:
        return self.__max
    
class Minimizer:
    def __init__(self) -> None:
        self.__min = float('inf')
    
    def join(self, *vals: List[float]) -> bool:
        val = min(vals)
        update_required = val < self.__min
        if update_required: self.__min = val
        return update_required
        
    def get(self) -> float:
        return self.__min
    
class Timer:
    def __init__(self) -> None:
        self.__dur = 0.0
    
    def __call__(self, fn: Callable, *args: Any, **kwds: Any) -> Any:
        start = time.time()
        res = fn(*args, **kwds)
        self.__dur += time.time() - start
        return res
    
    def get(self) -> float:
        return self.__dur

class Batcher:
    def __init__(self, items: Union[ndarray, int], batch_size: int) -> None:
        if isinstance(items, int):
            items = np.arange(items)
        self.__items = items
        self.__batch_size = batch_size

    def __shuffle(self) -> None:
        total_size = len(self.__items)
        self.__batch_num = math.ceil(total_size // self.__batch_size)
        self.__cur = 0
        self.__idx = np.arange(total_size)

        np.random.shuffle(self.__idx)
    
    def __iter__(self) -> Any:
        self.__shuffle()
        return self
    
    def __next__(self) -> ndarray:
        if self.__cur >= self.__batch_num:
            raise StopIteration
        cur = self.__cur
        size = self.__batch_size
        idx = self.__idx[cur * size: (cur + 1) * size]
        self.__cur += 1
        items = self.__items[idx]
        return items