#!/bin/bash
__doc__="""
Helper script to build wheels locally
"""
cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64  
