FROM python:3.7.7-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY protos protos

RUN apt-get update \
    && apt-get install make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf ~/.cache/pip

RUN make protobuf