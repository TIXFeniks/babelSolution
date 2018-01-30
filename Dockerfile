ARG repository
FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:jonathonf/python-3.6 && \ 
    apt-get update && \
    apt-get install -y --no-install-recommends python3.6

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN pip install tensorflow-gpu==1.4.0
