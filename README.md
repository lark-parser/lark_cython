# Lark-Cython

Cython plugin for Lark, reimplementing the LALR parser &amp; lexer for better performance on CPython.

**WIP**

Usage:

```python
from lark_cython import lark_cython

parser = Lark(grammar, _plugins=lark_cython.plugins)

# Use Lark as you usually would, with a huge performance boost
```

## Differences from Lark

- `Token` instances do not inherit from `str`. You must use the `value` attribute to get the string.

## Other caveats

- Postlexer isn't currently implemented

## Speed

In current benchmarks, lark-cython is about 50% to 80% faster than Lark.

We're still in the early stages, and in the future, lark-cython might go a lot faster.

## Other

License: MIT

Author: [Erez Shinan](https://github.com/erezsh/)

Special thanks goes to Datafold for commissioning the draft for lark-cython, and allowing me to relase it as open-source.