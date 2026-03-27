FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown

COPY . .

RUN mkdir -p data/source && \
    gdown 1qcCDp-2PkiYXWC3O43HQPPZpQkvwYOgQ && \
    mv sample_data_raw.csv data/source/transactions.csv

RUN python -m stage_04_ml.train_model
RUN python -m stage_04_ml.score_customers

EXPOSE 5000

CMD ["python", "app/app.py"]