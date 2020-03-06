FROM tensorflow/tensorflow:1.15.2-py3
WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
