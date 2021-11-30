FROM python:3.8.12-slim-buster

COPY artia /artia
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn artia.api:app --host 0.0.0.0 --port $PORT
#port request by google cloud run
