from src.solve_binary_optimization.util import binom
import math

batch_size = 1024 * 1024 * 20


print(binom(36, 18))

num_combinations = binom(36, 18)
num_batches = math.ceil(num_combinations / batch_size)
print(num_batches)
