##
## Neuralyzer Dev Dockerfile - Basic Jupyter Container
## based on minimal-notebook :
## https://github.com/jupyter/docker-stacks/tree/master/minimal-notebook
##
## Author: Michael Graber
##
## September 2015



FROM debian:jessie

MAINTAINER Michael Graber <michael.graber@gmail.com>
LABEL Description="Base jupyter container for the neuralyzer package."

USER root

# Install all OS dependencies for fully functional notebook server
# including dependencies for cvxpy installation
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
    openssl \
    libhdf5-dev \
    libatlas-base-dev \
    gfortran \
    gcc \
    git \
    vim \
    wget \
    build-essential \
    python-dev \
    ca-certificates \
    bzip2 \
    unzip \
    libsm6 \
    pandoc \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    supervisor \
    sudo \
    && apt-get clean


# Configure environment
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install conda
RUN echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-3.9.1-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-3.9.1-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda install --yes conda==3.14.1

# Install Jupyter notebook
RUN conda install --yes \
    'notebook=4.0*' \
    terminado \
    && conda clean -yt

# Configure container startup
EXPOSE 8888
CMD [ "start-notebook.sh" ]

# Add local files as late as possible to avoid cache busting
COPY start-notebook.sh /usr/local/bin/
RUN chmod 775 /usr/local/bin/start-notebook.sh

COPY notebook.conf /etc/supervisor/conf.d/
COPY jupyter_notebook_config.py /etc/skel/.jupyter/
