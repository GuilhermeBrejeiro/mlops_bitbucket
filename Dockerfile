FROM python:3.8.10

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CDM ["python", "main.py"]