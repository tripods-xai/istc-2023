# Latest as of 2022-11-09
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y python-pip && \
    apt-get install -y sudo

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY docker_requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && \ 
    pip uninstall -y setuptools && pip install setuptools && \
    pip install -r /opt/app/requirements.txt