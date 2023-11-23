__all__ = ['nvtx', 'nvtx_lambda', 'nvtx_with_repr']

import torch.cuda.nvtx


def nvtx(message):

  def wrapper(func):

    def wrapped(*args, **kwargs):
      with torch.cuda.nvtx.range(message):
        return func(*args, **kwargs)

    return wrapped

  return wrapper


def nvtx_lambda(factory):

  def wrapper(func):

    def wrapped(*args, **kwargs):
      with torch.cuda.nvtx.range(factory(*args, **kwargs)):
        return func(*args, **kwargs)

    return wrapped

  return wrapper


def nvtx_with_repr(func):
  return nvtx_lambda(lambda self, *args, **kwargs: repr(self))(func)
