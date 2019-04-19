FROM python:3.7-slim
MAINTAINER Edward Ross <edward@skeptric.com>

RUN apt update
RUN apt install -y gcc

ENV INSTALL_PATH /whatcar
RUN mkdir -p $INSTALL_PATH

WORKDIR $INSTALL_PATH

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

WORKDIR whatcar

CMD python serve.py

EXPOSE 80
