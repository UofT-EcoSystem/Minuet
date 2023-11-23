__all__ = ['registry', 'HasherRegistry']

import hashlib

from minuet.utils.registry import Registry


class HasherRegistry(Registry):

  def lookup(self, key: str, fallback=True, default=None):
    key = key.lower()
    return super(HasherRegistry, self).lookup(key, fallback, default)


registry = HasherRegistry()
"""
Registry for all hash methods.
"""

registry.register("md5", hashlib.md5)
registry.register("sha1", hashlib.sha1)
registry.register("sha224", hashlib.sha224)
registry.register("sha256", hashlib.sha256)
registry.register("sha384", hashlib.sha384)
registry.register("sha512", hashlib.sha512)
