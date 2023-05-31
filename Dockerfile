FROM python:3.10.1-buster

COPY main.py /main.py

RUN python -m pip install --upgrade pip
RUN pip install -r requirements_local.txt

ENTRYPOINT ["python", "/main.py"]
# RUN python /main.py
