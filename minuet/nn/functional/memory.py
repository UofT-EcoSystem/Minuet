__all__ = [
    'cuda_buffer_total_size', 'cuda_buffer_used_size', 'cuda_free_buffers',
    'cuda_set_buffer_growth', 'cuda_set_buffer_page_size',
    'cuda_preallocate_buffer', 'cuda_reset_error'
]

from minuet.nn.functional import _C


def cuda_buffer_total_size():
  return _C.cuda_buffer_total_size()


def cuda_buffer_used_size():
  return _C.cuda_buffer_total_size()


def cuda_free_buffers():
  return _C.cuda_free_buffers()


def cuda_preallocate_buffer(size: int):
  return _C.cuda_preallocate_buffer(size)


def cuda_set_buffer_growth(growth: float):
  return _C.cuda_set_buffer_growth(growth)


def cuda_set_buffer_page_size(growth: int):
  return _C.cuda_set_buffer_page_size(growth)


def cuda_reset_error():
  _C.cuda_reset_error()
