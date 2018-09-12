FROM tensorflow/tensorflow:latest-gpu-py3

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

#install scikit-image
RUN ${PIP} install scikit-image

#installing keras
RUN ${PIP} install keras

#installing pytorch
ARG PYTORCH_VERSION=torch-0.4.1
ARG PYTORCH_PYTHON_VERSION=cp35
ARG PYTORCH_CUDA_VERSION=cu92
RUN ${PIP} install http://download.pytorch.org/whl/${PYTORCH_CUDA_VERSION}/${PYTORCH_VERSION}-${PYTORCH_PYTHON_VERSION}-${PYTORCH_PYTHON_VERSION}m-linux_x86_64.whl
RUN ${PIP} install torchvision



