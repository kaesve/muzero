"""
"""
from dataclasses import dataclass
from typing import Union

__all__ = ['AverageMeter']


@dataclass
class AverageMeter(object):
    """
    Keep track of an uniform average and the last recorded value
    """
    val: float = 0.0
    avg: float = 0.0
    sum: int = 0
    count: int = 0

    def reset(self) -> None:
        """Set all values to zero"""
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: Union[int, float, complex], n: int = 1):
        """Update current data."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
