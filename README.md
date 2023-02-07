## Anime Recommender System[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

This project was carried out for the "Big Data Computing" exam, Master degree in Computer Science at Sapienza university - January 2023

<img src="logo.png" align="left" width="192px" height="192px"/>
<img align="left" width="0" height="192px" hspace="10"/>


With the advent of digital transformation, a low-code development platform that anyone can use to rapidly develop new ones is an essential tool for enabling your business to adapt and thrive.
Low-code development platforms make companies more productive and flexible by offering: efficiency, flexibility, responsiveness, iterability.

<br><br>
<br>

## Pre-requisites

The project was developed using python 3.6.7 with the following packages.
- Pandas
- Pillow
- PySpark
- Scikit_learn
- Streamlit
- Streamlit_card

Installation with pip:

```bash
pip install -r requirements.txt
```

## Getting Started
Open the terminal in you machine and run the following command to access the web application in your localhost.
```bash
streamlit run app.py
```

## Run on Docker
Alternatively you can build the Docker container and access the application at `localhost:8051` on your browser.
```bash
docker build --tag app:1.0 .
docker run --publish 8051:8051 -it app:1.0
```
## Files
- anime_reconmmendation_system.ipynb : Jupyter Notebook with all the workings including pre-processing, modelling and inference.
- app.py : Streamlit App script
- requirements.txt : pre-requiste libraries for the project
- models/ : cvs files (note: it does not contain animelist.csv because it is too heavy. It has to be downloaded manually
- notebooks/ : notebooks used for the project. One for the creation of recommendation systems and one for the data exploration phase
- pages/ : python file for the rating page for the web app

## Summary
The project aims to improve the user experience with anime content by assisting new users in discovering the anime world and helping existing users explore new options. The solution is a recommender system that employs two types of filtering: collaborative filtering and content-based filtering.

The dataset used is the MyAnimeList Database 2020, which contains 109 million reviews, 17,562 anime, and 325,772 unique users. The data was cleaned and explored, and some of the findings include the high number of unrated reviews, the top 10 highest average ratings and most reviewed anime, and the classification of anime by genres.

The popularity-based approach uses the 20 most popular anime to avoid the cold start problem and help new users. The collaborative filtering approach uses the ALS method to make recommendations based on the preferences of similar users. The performance of the model was evaluated using RMSE, MSE, and MAE.

The content-based approach with user profile creates personalized recommendations limited by user ratings and uses One Hot Encoding technique for genres. The content-based approach with cosine similarity uses cosine similarity to measure the similarity between two vectors. The performance was evaluated using precision and recall.

Finally, a web application was created to demonstrate the anime recommender system.

<br><br>
<br>
<br>
<br>

## Authors :thumbsup:

> Those who participated in the creation of the project are listed here

* [Alberto Cotumaccio](https://it.linkedin.com/in/alberto-cotumaccio-8b8443229?trk=people-guest_people_search-card)
* Giovanni Montobbio


## Acknowledgements

[Kaggle](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020), for providing the anime data.
[Streamlit](https://www.streamlit.io/), for the open-source library for rapid prototyping.
