FROM ubuntu:22.04
# FROM python:3.10.1-buster
RUN apt-get update && apt-get install -y python3.9 pip

COPY main.py /main.py
COPY requirements_local.txt /requirements_local.txt
COPY tests/supplements/optuna_test.py /optuna_test.py
COPY tests/supplements/study.pkl /study.pkl

RUN python3 -m pip install --upgrade pip
RUN pip install -r /requirements_local.txt

# ENTRYPOINT ["python", "/main.py"]
ENTRYPOINT ["pytest", "-v", "/optuna_test.py"]
# RUN python /main.py
