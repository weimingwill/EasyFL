FROM ubuntu:16.04
MAINTAINER Zhuang Weiming <wingalong@gmail.com>

RUN apt update
RUN apt -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.7
RUN apt-get update
RUN apt-get install -y wget python3-pip python3-dev libssl-dev libffi-dev bash

RUN mkdir /app
WORKDIR /app

RUN wget https://github.com/jwilder/docker-gen/releases/download/0.3.3/docker-gen-linux-amd64-0.3.3.tar.gz
RUN tar xvzf docker-gen-linux-amd64-0.3.3.tar.gz -C /usr/local/bin

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install etcd3

ADD . /app

ENV DOCKER_HOST unix:///var/run/docker.sock

CMD docker-gen -interval 10 -watch -notify "python3 /tmp/register.py" etcd.tmpl /tmp/register.py
