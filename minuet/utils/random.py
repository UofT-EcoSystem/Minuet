__all__ = ['random_points', 'batch_random_points']

import random

import numpy as np


def random_points(ndim, n, c_min, c_max, dtype=np.int32):
  tables = {}
  max_coords = pow(c_max - c_min + 1, ndim)
  if n > max_coords:
    raise ValueError(f"Cannot sample {n} points without replacement from a "
                     f"space with only {max_coords} possible points")

  points = []
  for i in range(n):
    number = random.randrange(i, max_coords)
    value = tables.get(number, number)
    tables[number] = tables.get(i, i)
    point = []
    for j in range(ndim):
      value, x = divmod(value, c_max - c_min + 1)
      point.append(x + c_min)
    points.append(point)
  return np.asarray(points, dtype=dtype)


def batch_random_points(ndim, n, batch_size, c_min, c_max, dtype=np.int32):
  batch = [
      random_points(ndim, n, c_min, c_max, dtype) for _ in range(batch_size)
  ]
  batch_dims = np.arange(0, batch_size + 1) * n
  batch = np.concatenate(batch, axis=0, dtype=dtype)
  return batch, batch_dims
