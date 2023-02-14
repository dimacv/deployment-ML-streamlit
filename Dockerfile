FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/dimacv/deployment-ML-streamlit.git

WORKDIR /app/deployment-ML-streamlit

RUN python -m pip install --upgrade pip  
RUN pip3 install -r requirements.txt

EXPOSE 8501

# Инструкция  HEALTHCHECKсообщает Docker, как протестировать контейнер, чтобы убедиться, что он все еще работает.
# Этот контейнер должен прослушивать порт Streamlit (по умолчанию) 8501:
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
