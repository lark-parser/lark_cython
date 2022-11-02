import re
from setuptools import find_packages, setup

__version__ ,= re.findall('__version__ = "(.*)"', open('lark_cython/__init__.py').read())

# python .\setup.py build_ext --inplace


# Delayed import; https://stackoverflow.com/questions/37471313/setup-requires-with-cython
try:
    from Cython.Build import cythonize
except ImportError:
     def cythonize(*args, **kwargs):
         from Cython.Build import cythonize
         return cythonize(*args, **kwargs)


def parse_description():
    """
    Parse the description in the README file
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.md')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            text = f.read()
        return text
    return ''

setup(
    name = "lark-cython",
    version = __version__,
    packages=find_packages(),

    ext_modules = cythonize('lark_cython/*.pyx'), # accepts a glob pattern
    requires = ['Cython'],
    install_requires = ['lark>=1.1.4', 'cython>=0.29.0', 'Cython>=0.29.0'],
    setup_requires=['Cython'],

    author = "Erez Shinan",
    author_email = "lark@erezsh.com",
    description = "A Lark plugin that optimizes LALR parsing using Cython",
    keywords = "Lark LALR parser optimized Cython",
    url = "https://github.com/lark-parser/lark_cython",
    long_description = parse_description(),
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    python_requires = '>=3.6',
    classifiers = [
        # List of classifiers available at:
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: MIT License',
        # Supported Python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
