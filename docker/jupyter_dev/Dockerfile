##
## Neuralyzer Dev Dockerfile
##
## Author: Michael Graber
##
## September 2015

# choose the correct image .. that contains a complete eups installation
# of all dependencies required by your webapp and the deployment dependency
# modWSGI
FROM michigraber/neuralyzer:base

MAINTAINER Michael Graber <michael.graber@gmail.com>
LABEL Description="Docker container to serve the neuralyzer package in an jupyter notebook."

# setup up the development workspace
WORKDIR /workspace/lib

# Get the dev-version of neuralyzer
RUN git clone https://github.com/michigraber/neuralyzer.git
WORKDIR neuralyzer 
RUN python setup.py develop

# Setting up the environment used by notebook.sh -> can be overwritten
ENV PEM_FILE /key.pem
# $PASSWORD will get `unset` within notebook.sh, turned into an IPython style hash
ENV PASSWORD pwd 
ENV USE_HTTP 0

# Add current files to / and set entry point.
#ADD . /workspace
ADD notebook.sh /notebook.sh
RUN chmod a+x /notebook.sh

WORKDIR /workspace
RUN mkdir notebooks
WORKDIR notebooks 

EXPOSE 8888

CMD ["/notebook.sh"]
