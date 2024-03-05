FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Optionally, set environment variables if needed
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install vim wget nano curl git git-lfs ca-certificates -y

RUN apt-get update
RUN apt-get install -y python3

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip

RUN pip3 install -i https://test.pypi.org/simple/ bitsandbytes
RUN pip3 install -q -U git+https://github.com/huggingface/accelerate.git

RUN pip3 install datasets==2.11.0 loralib sentencepiece git+https://github.com/huggingface/transformers.git gradio appdirs
RUN pip3 install -q -U git+https://github.com/huggingface/peft.git

RUN pip3 install scipy

WORKDIR /workspace

