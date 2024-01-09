FROM armswdev/tensorflow-arm-neoverse:latest

WORKDIR /app

ENV FLASK_APP=app.py
ENV TF_ENABLE_ONEDNN_OPTS=0 
ENV FLASK_ENV=development

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

#CMD ["python", "app.py"]
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]