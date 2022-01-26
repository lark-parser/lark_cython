# Lark-Cython

Cython plugin for Lark, reimplementing the LALR parser &amp; lexer for better performance on CPython.

**WIP**

Usage:

```python
from lark_cython import lark_cython

parser = Lark(grammar, _plugins=lark_cython.plugins)

# Use Lark as you usually would, with a huge performance boost
```
