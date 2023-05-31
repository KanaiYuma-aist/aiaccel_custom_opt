FROM python:3.10.1-buster

COPY main.py /main.py

# ENTRYPOINT ["python", "/main.py"]
RUN python /main.py
