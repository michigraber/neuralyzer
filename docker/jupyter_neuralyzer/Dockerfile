##
## Neuralyzer Dev Dockerfile
##
## Author: Michael Graber
##
## September 2015

FROM michigraber/neuralyzer:jupyter_minimal

MAINTAINER Michael Graber <michael.graber@gmail.com>
LABEL Description="Docker container to serve the neuralyzer package in an jupyter notebook."


USER root

# Install Python 3 science and data analysis packages
RUN conda install --yes \
    'ipywidgets=4.0*' \
    'pandas=0.16*' \
    'matplotlib=1.4*' \
    'scipy=0.15*' \
    'seaborn=0.6*' \
    'scikit-learn=0.16*' \
    'scikit-image=0.11*' \
    'sympy=0.7*' \
    'cython=0.22*' \
    'patsy=0.3*' \
    'statsmodels=0.6*' \
    'cloudpickle=0.1*' \
    'dill=0.2*' \
    'numba=0.20*' \
    'bokeh=0.9*' \
    pip \
    && conda clean -yt

# Install Python 2 packages and kernel spec
#RUN conda create -p $CONDA_DIR/envs/python2 python=2.7 \
    #'ipython=4.0*' \
    #'ipywidgets=4.0*' \
    #'pandas=0.16*' \
    #'matplotlib=1.4*' \
    #'scipy=0.15*' \
    #'seaborn=0.6*' \
    #'scikit-learn=0.16*' \
    #'scikit-image=0.11*' \
    #'sympy=0.7*' \
    #'cython=0.22*' \
    #'patsy=0.3*' \
    #'statsmodels=0.6*' \
    #'cloudpickle=0.1*' \
    #'dill=0.2*' \
    #'numba=0.20*' \
    #'bokeh=0.9*' \
    #pip \
    #pyzmq \
    #&& conda clean -yt
#RUN $CONDA_DIR/envs/python2/bin/python \
    #$CONDA_DIR/envs/python2/bin/ipython \
    #kernelspec install-self


# THE NEURALYZER DEVELOPMENT PACKAGES
# -----------------------------------------------------------------------------

# Get cvxpy using pip
RUN pip install cvxpy

# setup and install the development  
RUN mkdir /devlib

# Get the dev-version of neuralyzer
WORKDIR /devlib
RUN git clone https://github.com/michigraber/neuralyzer.git
WORKDIR neuralyzer 
RUN python setup.py develop

# Get the dev-version of scikit-learn
WORKDIR /devlib
RUN wget https://github.com/blink1073/tifffile/archive/v0.7.0.tar.gz
RUN tar xvfz v0.7.0.tar.gz
WORKDIR tifffile-0.7.0 
RUN python setup.py install

# Get the dev-version of scikit-learn
WORKDIR /devlib
RUN git clone https://github.com/scikit-learn/scikit-learn.git
WORKDIR scikit-learn
RUN python setup.py develop
