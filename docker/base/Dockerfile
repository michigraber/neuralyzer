##
## Neuralyzer Dev Dockerfile
##
## Author: Michael Graber
##
## September 2015

FROM ubuntu:14.04

MAINTAINER Michael Graber <michael.graber@gmail.com>
LABEL Description="Base container for the neuralyzer package."

ENV DEBIAN_FRONTEND noninteractive

# Install packages
RUN apt-get -y update && apt-get install -y wget openssl libhdf5-dev libatlas-base-dev gfortran python-dev python-nose git g++

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8

RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales

# Install and setup minimal Anaconda Python distribution
RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /anaconda && rm miniconda.sh
ENV PATH /anaconda/bin:$PATH

# For image inheritance.
ONBUILD ENV PATH /anaconda/bin:$PATH

# Install some essential data science packages using conda
RUN conda install scipy numpy scikit-learn scikit-image pyzmq nose readline pandas matplotlib seaborn dateutil ipython-notebook pip hdf5

# Get cvxpy using pip
RUN pip install cvxpy
RUN nosetests cvxpy

# setup up the development workspace
RUN mkdir /workspace
RUN mkdir /workspace/lib
WORKDIR /workspace/lib

# Get the dev-version of scikit-learn
RUN mkdir external
WORKDIR external
RUN wget https://github.com/blink1073/tifffile/archive/v0.7.0.tar.gz
RUN tar xvfz v0.7.0.tar.gz
WORKDIR tifffile-0.7.0 
RUN python setup.py install

# Get the dev-version of scikit-learn
WORKDIR /workspace/lib/external
RUN git clone https://github.com/scikit-learn/scikit-learn.git
WORKDIR scikit-learn
RUN python setup.py develop

WORKDIR /
