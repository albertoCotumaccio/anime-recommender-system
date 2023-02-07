import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from streamlit_card import card
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from sklearn.preprocessing import MinMaxScaler

from pyspark.sql import functions as F


st.set_page_config(layout="wide")

st.title('Give ratings to your favourite animes')

def load_dataframe(name, spark, separator):
   return spark.read.load(name,  format="csv",  sep= separator,  inferSchema="true",  header="true" )
   


conf = SparkConf().set("spark.ui.port", "4051").set('spark.executor.memory', '12G').set('spark.driver.memory', '45G').set('spark.driver.maxResultSize', '10G').set('spark.worker.memory', '12G')
spark = SparkSession.builder.appName("Recommendation").getOrCreate()

def clear_multi():
    st.session_state.multiselect = []
    return


user_input = []
def give_ratings(list_ratings):
   with st.form("Form"):
      for i in range(len(list_ratings)):
         diz = dict()
         box_val = st.markdown('**' + list_ratings[i] + '**')
         slider_val = st.slider(label='Select rating for ' + list_ratings[i], min_value=1, max_value=10, key=1000+i)
         #diz["user_id"] = -1
         #diz["Name"] =  list_ratings[i]
         diz["MAL_ID"] = df.filter(df.Name == list_ratings[i]).collect()[0][0]
         diz["Name"] = df.filter(df.Name == list_ratings[i]).collect()[0][1]
         diz["rating"] =  slider_val
         user_input.append(diz)
      submitted = st.form_submit_button('Submit')
      if submitted:
         run_model()

         
def run_model():
   #Inserire codice della funzione del content filtering creando user input con le informazioni degli anime
   inputAnime = pd.DataFrame(user_input)
   #st.dataframe(pdf)
   #inserire codice
   pdf['Genres'] = pdf.Genres.str.split(',')
   animeWithGenre_pdf = df.select(df.MAL_ID, df.Name).toPandas()
   for index, row in pdf.iterrows():
      for genre in row['Genres']:
            clean_genre = genre.strip()
            animeWithGenre_pdf.at[index,clean_genre] = 1

   animeWithGenre_pdf = animeWithGenre_pdf.fillna(0)
   st.write("Your 20 recommended anime")
   st.dataframe(content_recommendation(inputAnime,20,animeWithGenre_pdf))

def content_recommendation(user_input,n, animeWithGenre_pdf):
  userAnime = animeWithGenre_pdf[animeWithGenre_pdf['MAL_ID'].isin(user_input['MAL_ID'].tolist())]
  #Resetting the index to avoid future issues
  userAnime = userAnime.reset_index(drop=True)
  #Dropping unnecessary issues due to save memory and to avoid issues
  userGenreTable = userAnime.drop('MAL_ID', 1).drop('Name', 1)
  userProfile = userGenreTable.transpose().dot(user_input['rating'])
  #Now let's get the genres of every anime in our original dataframe
  genreTable = animeWithGenre_pdf.set_index(animeWithGenre_pdf['MAL_ID'])
  #And drop the unnecessary information
  genreTable = genreTable.drop('MAL_ID', 1).drop('Name', 1)
  recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
  #Sort our recommendations in descending order
  recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
  # Remove animes of the user input from the finale recommendation table
  recommendationTable_df = recommendationTable_df.drop(user_input["MAL_ID"].to_list())  
  #The final recommendation table
  return pdf.loc[pdf['MAL_ID'].isin(recommendationTable_df.head(n).keys())]


content_df = load_dataframe("models/anime.csv", spark, ",")
content2_df = load_dataframe("models/anime2.csv", spark, "\t")
content2_df = content2_df.select(content2_df.anime_id, content2_df.main_pic)
df = content_df.join(content2_df,
               content_df.MAL_ID == content2_df.anime_id, 
               "left")
df = df.withColumn("Score", df["Score"].cast('double'))
df = df.withColumn('main_pic', F.when(df.main_pic.isNull(), F.lit('missing_image.jpeg'))
               .otherwise(df.main_pic))
df = df.filter(df.Genres != "Unknown")
df = df.filter( (df.MAL_ID != 37490) & (df.MAL_ID != 31630) & (df.MAL_ID != 16187) )
df = df.dropDuplicates(["Name"]) 
pdf = df.toPandas()



list_ratings = st.multiselect('Choose animes you have watched', df.select(['Name',]).rdd.flatMap(lambda x: x).collect(),key="multiselect")

if list_ratings:
   give_ratings(list_ratings)






