# Lark-Cython

Cython plugin for [Lark](https://github.com/lark-parser/lark), reimplementing the LALR parser &amp; lexer for better performance on CPython.

Install:

```python
pip install lark-cython
```

Usage:

```python
import lark_cython

parser = Lark(grammar, parser="lalr", _plugins=lark_cython.plugins)

# Use Lark as you usually would, with a huge performance boost
```

See the [examples](https://github.com/lark-parser/lark_cython/tree/master/examples) for more.


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