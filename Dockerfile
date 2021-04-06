FROM python:3.7-slim
MAINTAINER Edward Ross <edward@skeptric.com>

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN apt update
RUN apt install -y gcc

ENV APP_HOME /whatcar
RUN mkdir -p $APP_HOME

WORKDIR $APP_HOME

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

WORKDIR whatcar

CMD python serve.py $PORT