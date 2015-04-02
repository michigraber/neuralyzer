
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "neuralyzer",
    version = "0.0.1",
    author = "Michael Graber",
    author_email = "michigraber@gmail.com",
    description = ("A neuroscience data analysis tool suite."),
    license = "MIT",
    keywords = "neuroscience, data analysis, image processing",
    packages=[
        'neuralyzer',
        'neuralyzer.io',
        'neuralyzer.io.plugins',
        'neuralyzer.utils',
        ],
    scripts=['scripts/build_tiff_stack.py',],
    long_description=read('README.rst'),
)
