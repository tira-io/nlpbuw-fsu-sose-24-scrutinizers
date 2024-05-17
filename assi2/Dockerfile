# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip3 install pandas
ADD ngramkf.py /code/ngramkf.py
ADD run.py /code/run.py
RUN sudo apt-get install unzip
# trained model exceeds github file size limit, so we download it from the webis server
RUN wget https://files.webis.de/teaching/nlp-ss24-weimar/assignment-2-group-4/model.joblib -O /code/model.joblib
ADD vectorizer.joblib /code/vectorizer.joblib
# ADD ./knn_classifier/. /code/knn_classifier/

ENTRYPOINT ["python", "/code/run.py"]
