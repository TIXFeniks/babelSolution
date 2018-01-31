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
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.6 && \
    apt-get install -y python3.6-dev

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN pip install tensorflow-gpu==1.4.0
RUN pip install Cython
RUN pip install tqdm pandas matplotlib fasttext keras sklearn

RUN mkdir /nmt
WORKDIR /nmt

RUN mkdir /nmt/ext_libs
RUN git clone https://github.com/moses-smt/mosesdecoder.git /nmt/ext_libs/mosesdecoder
RUN git clone https://github.com/rsennrich/subword-nmt.git /nmt/ext_libs/subword-nmt
RUN apt-get install -y python3-tk

COPY . /nmt
