"""
Basic calculator
================

A simple example of a REPL calculator

This example shows how to write a basic calculator with variables.

Adapted from the Lark Calculator example (lark/examples/calc.py).

Main differences from Lark's example code:

- We use the _plugins option to override Lark's internal lexer+parser implementation

- Since Tokens don't inherit from str, we have to explicitly use "token.value".
"""
from lark import Lark, Transformer, v_args
import lark_cython

try:
    input = raw_input   # For Python2 compatibility
except NameError:
    pass


calc_grammar = """
    ?start: sum
          | NAME "=" sum    -> assign_var

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | NAME             -> var
         | "(" sum ")"

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""


@v_args(inline=True)    # Affects the signatures of the methods
class CalculateTree(Transformer):
    from operator import add, sub, mul, truediv as div, neg

    def number(self, t: lark_cython.Token) -> float:
        return float(t.value)

    def NAME(self, t: lark_cython.Token) -> str:
        return t.value

    def __init__(self):
        self.vars = {}

    def assign_var(self, name, value):
        self.vars[name] = value
        return value

    def var(self, name):
        try:
            return self.vars[name]
        except KeyError:
            raise Exception("Variable not found: %s" % name)


calc_parser = Lark(calc_grammar, parser='lalr', transformer=CalculateTree(), _plugins=lark_cython.plugins)
calc = calc_parser.parse


def main():
    while True:
        try:
            s = input('> ')
        except EOFError:
            break
        print(calc(s))


def test():
    print(calc("a = 1+2"))
    print(calc("1+a*-3"))


if __name__ == '__main__':
    # test()
    main()
