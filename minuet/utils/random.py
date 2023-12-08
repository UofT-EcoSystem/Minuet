__all__ = ['random_unique_points']

import random

import numpy as np


def random_unique_points(ndim: int,
                         n: int,
                         c_min: int,
                         c_max: int,
                         dtype=np.int32):
  r"""
  Generate random coordinates without duplicates

  Args:
    ndim: the dimension of each coordinate
    n: the number of points to be generated
    c_min: the minimum coordinate
    c_max: the maximum coordinate
    dtype: the coordinate data type

  Returns:
    a numpy array consists of generated coordinates
  """
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
