# PySaDML

## Presentation

This repository contains the code for our project **PySaDML**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to ** to detect whether the sound emitted by a target machine is normal or abnormal **

This project was developed by the following team :

- Laurent Noyelle ([LinkedIn](http://linkedin.com/in/lnoyelle))
- Steve Minlo ([LinkedIn](www.linkedin.com/in/steve-minlo-b3a844b9))
- Mike Guidy ([GitHub](https://github.com/Dwaynekiller) / [LinkedIn](https://www.linkedin.com/in/mike-eddie-g-04404144/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit_app
docker build -t streamlit-app .
```

You can then run the container using :

```shell
docker run --name streamlit-app -p 8501:8501 streamlit-app
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
