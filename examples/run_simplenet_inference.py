import argparse
from typing import Optional

import numpy as np
import torch
import minuet


class SimplePCNet(torch.nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv1 = minuet.nn.SparseConv3d(
        in_channels=4,
        out_channels=32,
        kernel_size=3,
    )
    self.conv2 = minuet.nn.SparseConv3d(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=2,
    )
    self.conv3 = minuet.nn.SparseConv3d(
        in_channels=32,
        out_channels=32,
        kernel_size=3,
    )

  def forward(self, x: minuet.SparseTensor):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = minuet.nn.functional.global_avg_pool(x)
    return x


def make_dummy_point_cloud(
    batch_size: Optional[int],
    num_points: int,
    num_features: int = 4,
    # Minuet always requires sorted coordinates
    ordered: bool = True,
    c_min: int = 0,
    c_max: int = 100):
  if batch_size is None:
    coordinates = minuet.utils.random.random_unique_points(ndim=3,
                                                           n=num_points,
                                                           c_min=c_min,
                                                           c_max=c_max)
    coordinates = torch.tensor(coordinates).cuda()
    batch_dims = None
  else:
    coordinates = [
        minuet.utils.random.random_unique_points(ndim=3,
                                                 n=num_points,
                                                 c_min=c_min,
                                                 c_max=c_max)
        for _ in range(batch_size)
    ]
    batch_dims = [0]
    batch_dims.extend([len(c) for c in coordinates])
    batch_dims = np.cumsum(np.asarray(batch_dims))

    coordinates = torch.concat([torch.tensor(c) for c in coordinates])
    coordinates = coordinates.cuda()
    batch_dims = torch.tensor(batch_dims, device=coordinates.device)

  features = torch.randn(len(coordinates),
                         num_features,
                         device=coordinates.device)

  if ordered:
    index = minuet.nn.functional.arg_sort_coordinates(coordinates,
                                                      batch_dims=batch_dims)
    coordinates = coordinates[index]

    # Don't forget to permute your features
    # It doesn't matter for dummy inputs though
    features = features[index]

  return minuet.SparseTensor(features=features,
                             coordinates=coordinates,
                             batch_dims=batch_dims)


def main(args):
  tuning_data = [
      make_dummy_point_cloud(num_points=args.num_points,
                             batch_size=args.batch_size) for _ in range(5)
  ]

  net = SimplePCNet().cuda()
  net.eval()

  cache = minuet.nn.KernelMapCache(ndim=3, dtype=torch.int32, device="cuda:0")
  minuet.set_kernel_map_cache(module=net, cache=cache)

  # Autotuning is optional but it is better for performance
  minuet.autotune(net, cache, data=tuning_data)

  # At the current moment, Minuet does not support training
  with torch.no_grad():
    for i in range(10):
      # Before each different input the model cache must be reset
      # Note that the minuet.autotune may touch the cache as will
      cache.reset()
      dummy_input = make_dummy_point_cloud(num_points=args.num_points,
                                           batch_size=args.batch_size)
      print(net(dummy_input))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size",
                      "-B",
                      type=int,
                      default=None,
                      help="batch size for inference")
  parser.add_argument("--num_points",
                      "-N",
                      type=int,
                      required=True,
                      help="number of points for random generated point cloud")
  main(parser.parse_args())
