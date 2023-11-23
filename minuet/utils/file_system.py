__all__ = [
    'get_file_extension', 'get_filename', 'get_file_basename', 'get_directory',
    'unlink', 'as_file_object', 'get_relative_path', 'ensure_directory',
    'get_absolute_path', 'is_subdirectory', 'simplify_path', 'get_path_dirs',
    'get_file_checksum', 'is_identical_path'
]

import contextlib
import os
import shutil
from typing import IO, Union

import tqdm

from minuet.utils import hashlib, cli


def get_directory(path):
  return os.path.dirname(path)


def get_filename(path):
  return os.path.basename(path)


def get_file_basename(path):
  return get_filename(os.path.splitext(path)[0])


def get_file_extension(path):
  return os.path.splitext(path)[1]


def is_identical_path(a,
                      b,
                      use_real_path: bool = False,
                      use_user_path: bool = True,
                      use_vars_path: bool = True):
  a = get_absolute_path(a,
                        use_real_path=use_real_path,
                        use_user_path=use_user_path,
                        use_vars_path=use_vars_path)
  b = get_absolute_path(b,
                        use_real_path=use_real_path,
                        use_user_path=use_user_path,
                        use_vars_path=use_vars_path)
  return a == b


def unlink(path, force: bool = False):
  if os.path.exists(path):
    if os.path.isdir(path):
      shutil.rmtree(path, ignore_errors=force)
    else:
      os.remove(path)


@contextlib.contextmanager
def as_file_object(instance: Union[str, IO], **kwargs):
  if isinstance(instance, str):
    with open(instance, **kwargs) as handle:
      yield handle
  else:
    yield instance


def get_relative_path(target,
                      source='.',
                      use_real_path: bool = False,
                      use_user_path: bool = True,
                      use_vars_path: bool = True):
  target = get_absolute_path(target,
                             use_real_path=use_real_path,
                             use_user_path=use_user_path,
                             use_vars_path=use_vars_path)
  source = get_absolute_path(source,
                             use_real_path=use_real_path,
                             use_user_path=use_user_path,
                             use_vars_path=use_vars_path)
  return os.path.relpath(target, source)


def expand_path(path, use_user_path: bool = True, use_vars_path: bool = True):
  if use_user_path:
    path = os.path.expanduser(path)
  if use_vars_path:
    path = os.path.expandvars(path)
  return path


def get_absolute_path(path,
                      use_real_path: bool = False,
                      use_user_path: bool = True,
                      use_vars_path: bool = True):
  path = expand_path(path,
                     use_user_path=use_user_path,
                     use_vars_path=use_vars_path)
  return os.path.realpath(path) if use_real_path else os.path.abspath(path)


def is_subdirectory(path: str,
                    parent_path: str = ".",
                    use_real_path: bool = False,
                    use_user_path: bool = True,
                    use_vars_path: bool = True):
  path = get_absolute_path(path,
                           use_real_path=use_real_path,
                           use_user_path=use_user_path,
                           use_vars_path=use_vars_path)
  parent_path = get_absolute_path(parent_path,
                                  use_real_path=use_real_path,
                                  use_user_path=use_user_path,
                                  use_vars_path=use_vars_path)
  return path.startswith(parent_path)


def simplify_path(target,
                  source='.',
                  use_real_path: bool = False,
                  use_user_path: bool = True,
                  use_vars_path: bool = True):
  relative_path = get_relative_path(target,
                                    source,
                                    use_real_path=use_real_path,
                                    use_user_path=use_user_path,
                                    use_vars_path=use_vars_path)
  absolute_path = get_absolute_path(target,
                                    use_real_path=use_real_path,
                                    use_user_path=use_user_path,
                                    use_vars_path=use_vars_path)
  if len(relative_path) < len(absolute_path):
    return relative_path
  else:
    return absolute_path


def ensure_directory(path, create_if_not_exist=True):
  if not os.path.exists(path):
    if create_if_not_exist:
      os.makedirs(path)
    else:
      raise RuntimeError(f"Directory {path} does not exist")
  elif not os.path.isdir(path):
    raise RuntimeError(f"Path {path} is not a directory")
  return path


def get_path_dirs(path):
  result = []
  while path:
    parent_path, folder = os.path.split(path)
    result.append(folder)
    if path == parent_path:
      break
    path = parent_path

  result.reverse()
  return result


def get_file_checksum(path,
                      checksum_type: str,
                      chunk_size: int = 1024 * 1024,
                      show_progress_bar: bool = True):
  hasher = hashlib.registry.lookup(checksum_type, fallback=False)
  if hasher is None:
    raise ValueError(f"Unsupported checksum {checksum_type}")

  with as_file_object(path, mode="rb") as reader:
    bar = None
    if show_progress_bar:
      num_total_bytes = os.fstat(reader.fileno()).st_size
      header = cli.CLIColorFormat(bold=True)
      header = header.colored(f"Calculating {checksum_type.upper()}")
      colored_path = cli.CLIColorFormat(underline=True, color="green")
      colored_path = colored_path.colored(simplify_path(path))
      bar = tqdm.tqdm(total=num_total_bytes,
                      unit='B',
                      unit_scale=True,
                      leave=False,
                      miniters=1,
                      desc=f"{header} {colored_path}",
                      dynamic_ncols=True)
    try:
      hasher = hasher()
      while True:
        chunk = reader.read(chunk_size)
        if not chunk:
          break
        hasher.update(chunk)
        if bar is not None:
          bar.update(len(chunk))
      return hasher.hexdigest()
    finally:
      if bar is not None:
        bar.close()
