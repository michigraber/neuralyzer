#!/usr/bin/env python

from distutils.core import setup

setup(name='neuralyzer',
      version='0.0.1',
      description='A neuroscience imaging toolset.',
      author='Michael Graber',
      author_email='michael.graber@gmail.com',
      packages=[
          'neuralyzer',
          'neuralyzer.io',
          'neuralyzer.io.plugins',
          'neuralyzer.utils',
          ],
      scripts=['scripts/build_tiff_stack.py',]
     )
