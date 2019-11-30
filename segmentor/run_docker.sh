docker run \
    --runtime=nvidia \
    --rm \
    -it \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${CANCER_DATA_DIR}:/data \
    -v ${CANCER_WORKDIR}:/cancer_workdir \
    unet:latest bash