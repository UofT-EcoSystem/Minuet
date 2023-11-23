__all__ = [
    'BaseRegistry', 'CallbackRegistry', 'Registry', 'RegistryGroup',
    'DefaultRegistry'
]

import collections
import functools


class BaseRegistry(object):
  __FALLBACK_KEY__ = '__fallback__'

  def __init__(self):
    self._init_registry()

  def _init_registry(self):
    self._registry = dict()

  @property
  def fallback(self):
    return self._registry.get(self.__FALLBACK_KEY__, None)

  def set_fallback(self, value):
    self._registry[self.__FALLBACK_KEY__] = value
    return self

  def register(self, key, value):
    self._registry[key] = value
    return self

  def unregister(self, key):
    return self._registry.pop(key)

  def has(self, key):
    return key in self._registry

  def lookup(self, key, fallback=True, default=None):
    if fallback:
      fallback_value = self._registry.get(self.__FALLBACK_KEY__, default)
    else:
      fallback_value = default
    return self._registry.get(key, fallback_value)

  def keys(self):
    return list(self._registry.keys())

  def values(self):
    return list(self._registry.values())

  def items(self):
    return list(self._registry.items())


Registry = BaseRegistry


class DefaultRegistry(BaseRegistry):
  __base_class__ = dict

  def _init_registry(self):
    base_class = type(self).__base_class__
    self._registry = collections.defaultdict(base_class)

  def lookup(self, key, fallback=False, default=None):
    assert fallback is False and default is None
    return self._registry.get(key)


class CallbackRegistry(Registry):

  def _init_registry(self):
    super(CallbackRegistry, self)._init_registry()
    self._super_callback = None

  @property
  def super_callback(self):
    return self._super_callback

  def set_super_callback(self, callback):
    self._super_callback = callback
    return self

  @property
  def fallback_callback(self):
    return self.fallback

  def set_fallback_callback(self, callback):
    return self.set_fallback(callback)

  def register(self, key, value, *args, **kwargs):
    if args or kwargs:
      value = functools.partial(value, *args, **kwargs)
    super().register(key, value)

  def dispatch(self, key, *args, **kwargs):
    if self._super_callback is not None:
      return self._super_callback(key, *args, **kwargs)
    return self.dispatch_direct(key, *args, **kwargs)

  def dispatch_direct(self, key, *args, **kwargs):
    callback = self.lookup(key, fallback=False)
    if callback is None:
      if self.fallback_callback is None:
        raise ValueError(f'Unknown callback entry "{key}"')
      return self.fallback_callback(*args, **kwargs)
    return callback(*args, **kwargs)


class RegistryGroup(object):
  __base_class__ = Registry

  def __init__(self):
    self._init_registry_group()

  def _init_registry_group(self):
    base_class = type(self).__base_class__
    self._registry_group = collections.defaultdict(base_class)

  def register(self, registry_name, key, value):
    return self._registry_group[registry_name].register(key, value)

  def lookup(self, registry_name, key, fallback=True, default=None):
    return self._registry_group[registry_name].lookup(key, fallback, default)
