import os
import torch
import packaging.version

try:
  import pybind11
  print(os.path.join(os.path.dirname(os.path.abspath(pybind11.__file__)), "share", "cmake"))
except ImportError:
  print(None)

print(torch.utils.cmake_prefix_path)
print(torch.cuda.is_available())
print(packaging.version.parse(torch.version.cuda).major)

try:
  import nvidia.cublas
  print(nvidia.cublas.__path__[0])
except ImportError:
  print(None)
