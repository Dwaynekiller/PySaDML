# À construire depuis la racine du dépôt (l'app importe notebooks/include) :
#   docker build -f streamlit_app/Dockerfile -t pysadml-app .
FROM python:3.13-slim

# libsndfile est nécessaire à librosa / soundfile pour lire les .wav
RUN apt-get update     && apt-get install -y --no-install-recommends libsndfile1     && rm -rf /var/lib/apt/lists/*

WORKDIR /repo/streamlit_app

COPY streamlit_app/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY streamlit_app/ /repo/streamlit_app/
COPY notebooks/include/ /repo/notebooks/include/

# Les données (/repo/data) et les modèles (/repo/models) ne sont pas copiés dans
# l'image : ils doivent être montés au lancement (voir README).
EXPOSE 8501

CMD [ "streamlit", "run", "app.py", "--server.address=0.0.0.0" ]
