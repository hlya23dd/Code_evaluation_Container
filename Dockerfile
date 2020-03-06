FROM tensorflow/tensorflow:1.10.0-py3
WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
