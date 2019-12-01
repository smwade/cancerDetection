FROM nvcr.io/nvidia/tensorflow:19.11-tf1-py3

ADD . /static
WORKDIR /static

ENV CANCER_DATA_PATH=/workspace/sean/data
ENV CANCER_WORKDIR=/workspace/sean/cancerDetection

RUN ./setup.sh
WORKDIR /workspace
