import timeit

import torch

# Create a tensor
x = torch.randn(1000, 1000)

# Time the overhead of mul_(1)
mul_one_time = timeit.timeit(lambda: x, number=100000)

# Time the overhead of add_(0)
add_zero_time = timeit.timeit(lambda: x.add_(0), number=100000)

print(f"Overhead of mul_(1): {mul_one_time:.6f} seconds")
print(f"Overhead of add_(0): {add_zero_time:.6f} seconds")
