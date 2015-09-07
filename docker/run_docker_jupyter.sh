docker run -d \
    -p 8888:8888 \
    --hostname=NEURALYZER_DOCKER_DEV \
    -v /some/path/to/workdir:/home/jovyan/work \
    -v /some/data/path:/data \
    -e "PASSWORD=michael" \
    -e "GRANT_SUDO=1" \
    #-e "NB_UID=1000" \
    michigraber/neuralyzer:jupyter_dev
