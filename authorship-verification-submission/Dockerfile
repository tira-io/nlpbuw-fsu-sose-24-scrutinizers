# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

WORKDIR /app

RUN pip install langid pandas tqdm tira-rest-api-client

ADD assi2.py /app/assi2.py

ENTRY ["python", "assi2.py"]
