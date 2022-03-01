import re
from setuptools import find_packages, setup

__version__ ,= re.findall('__version__ = "(.*)"', open('lark_cython/__init__.py').read())

# python .\setup.py build_ext --inplace  

from distutils.core import setup


# Delayed import; https://stackoverflow.com/questions/37471313/setup-requires-with-cython
try:
    from Cython.Build import cythonize
except ImportError:
     def cythonize(*args, **kwargs):
         from Cython.Build import cythonize
         return cythonize(*args, **kwargs)

setup(
    name = "lark-cython",
    version = __version__,
    packages=find_packages(),

    ext_modules = cythonize('lark_cython/*.pyx'), # accepts a glob pattern
    requires = ['Cython'],
    install_requires = ['lark>=1.1.2', 'cython>=0.29.0', 'Cython>=0.29.0'],
    setup_requires=['Cython'],

    author = "Erez Shinan",
    author_email = "lark@erezsh.com",
    description = "A Lark plugin that optimizes LALR parsing using Cython",
    keywords = "Lark LALR parser optimized Cython",
    url = "https://github.com/lark-parser/lark_cython",
)