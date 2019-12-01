FROM nvcr.io/nvidia/tensorflow:19.11-tf1-py3

ADD . /workspace/cancer
WORKDIR /workspace/cancer

ENV CANCER_DATA_PATH=/workspace/sean/data
ENV CANCER_WORKDIR=/workspace/cancer

RUN ./setup.sh
