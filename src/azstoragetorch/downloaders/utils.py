import math
from typing import List, Tuple


def get_partitioned_reads(
    offset: int,
    length: int,
    partition_size: int
) -> List[Tuple[int, int]]:
    end = offset + length
    num_partitions = math.ceil(length / partition_size)
    partitions = []
    for i in range(num_partitions):
        start = offset + i * partition_size
        if start >= end:
            break
        size = min(partition_size, end - start)
        partitions.append((start, size))
    return partitions