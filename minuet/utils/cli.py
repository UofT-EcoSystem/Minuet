__all__ = ['CLIColorFormat']

import termcolor


class CLIColorFormat(object):

  def __init__(self,
               color=None,
               background=None,
               bold: bool = False,
               dark: bool = False,
               blink: bool = False,
               underline: bool = False,
               reverse: bool = False,
               concealed: bool = False):
    self._color = color
    self._background = background
    self._attributes = {
        'bold': bold,
        'dark': dark,
        'blink': blink,
        'reverse': reverse,
        'underline': underline,
        'concealed': concealed
    }

  @property
  def color(self):
    return self._color

  @property
  def background(self):
    return self._background

  @property
  def attributes(self):
    return [k for k, v in self._attributes.items() if v]

  def set_attribute(self, attribute, inuse=True):
    assert attribute in self._attributes
    self._attributes[attribute] = inuse
    return self

  def colored(self, text):
    return termcolor.colored(text, self._color, self._background,
                             self.attributes)
