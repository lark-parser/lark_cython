import re
from setuptools import setup

__version__ ,= re.findall('__version__ = "(.*)"', open('lark_cython/__init__.py').read())

# python .\setup.py build_ext --inplace  

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "lark-cython",
    version = "0.0.1",

    ext_modules = cythonize('lark_cython/*.pyx'), # accepts a glob pattern
    install_requires = ['lark', 'cython'],

    author = "Erez Shinan",
    author_email = "lark@erezsh.com",
    description = "A Lark plugin that optimizes LALR parsing using Cython",
    keywords = "Lark LALR parser optimized Cython",
)