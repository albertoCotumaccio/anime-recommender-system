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

#APPUNTI: 



def load_dataframe(name, spark, separator):
   return spark.read.load(name,  format="csv",  sep= separator,  inferSchema="true",  header="true" )
   

st.set_page_config(layout="wide")


def page_switcher(page, key):
   st.experimental_set_query_params(name=key)
   st.session_state.runpage = page

   
conf = SparkConf().set('spark.executor.memory', '12G').set('spark.driver.memory', '45G')


# create the context
#sc = pyspark.SparkContext(conf=conf)


spark = SparkSession.builder.getOrCreate()

spark = SparkSession.builder.appName("Recommendation").getOrCreate()



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


def main():
   #deleting duplicated names (5 anime)
   

   popularity_df = df.select(df.MAL_ID, df.Name, df.Score, df.Members, df.Favorites, df.main_pic)
   popularity_pdf = popularity_df.toPandas()
   scaler=MinMaxScaler()
   popularity_pdf[['Score','Members','Favorites']]=scaler.fit_transform(popularity_pdf[['Score','Members','Favorites']])
   popularity_pdf['Weighted_score']=popularity_pdf['Score']*0.5 + popularity_pdf['Members']*0.25 + popularity_pdf['Favorites']*0.25
   popularity_pdf = popularity_pdf.sort_values(by= 'Weighted_score',ascending=False).head(20)
   

   st.experimental_set_query_params(
    name="main",
   )



   li = popularity_pdf['main_pic'].apply(pd.Series).stack().drop_duplicates().tolist()
   captions = popularity_pdf['Name'].apply(pd.Series).stack().drop_duplicates().tolist()

   l = content_df.select('Name').rdd.flatMap(lambda x: (x)).collect()
   
   

  
   for i in range(1,6): # number of rows in your table! = 2
      cols = st.columns(4) # number of columns in each row! = 2
      # first column of the ith row
      for j in range(4):
         cols[j].image(li[0] , use_column_width=True, caption=captions[0], width =5)
 
         cols[j].button(captions[0],on_click=page_switcher,args=(details, captions[0]))
         li.pop(0)
         captions.pop(0)

   



def details():
   
   btn = st.button("go back")
   key = st.experimental_get_query_params()['name'][0]
   st.title(key)
   
   

   #print(df.select(df.main_pic).where(df.Name==key).collect())
   anime_df = df.filter(df.Name == key)
   anime_pdf = anime_df.toPandas()
   # anime pdf input per content2
   st.image((anime_pdf.main_pic).apply(pd.Series).stack().drop_duplicates().tolist()[0]) # Get anime image link
   st.write("Genres: "+ (anime_pdf.Genres).apply(pd.Series).stack().drop_duplicates().tolist()[0])
   
   
   
   if btn :
      
      st.experimental_set_query_params(name="main")
      st.session_state.runpage = main  
      st.experimental_rerun()

   st.title('Similar animes you might be interested in')
   content2(df.toPandas(), key)


def content2(content_pdf,name):
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import linear_kernel


   tf = TfidfVectorizer(analyzer='word', stop_words='english')
   tfidf_matrix = tf.fit_transform(content_pdf['Genres'])
   cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
   titles = content_pdf[['Name','main_pic']]
   indices = pd.Series(content_pdf.index, index=content_pdf['Name'])
   a = anime_recommendations(name, cosine_sim, indices,titles).head(21)
   a = a[a.Name != name].head(20)


   li = a['main_pic'].apply(pd.Series).stack().tolist()
   captions = a['Name'].apply(pd.Series).stack().drop_duplicates().tolist()

   for i,el in enumerate(li):
      print(i,el,"\n")



   for i in range(1,6): # number of rows in your table! = 2
      cols = st.columns(4) # number of columns in each row! = 2
      # first column of the ith row
      for j in range(4):
         cols[j].image(li[0] , use_column_width=True, caption=captions[0], width =5)
         captions.pop(0)
         li.pop(0)
      


def anime_recommendations(title, cosine_sim, indices,titles):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:22]
    anime_indices = [i[0] for i in sim_scores]
    return titles.iloc[anime_indices]




if __name__ == '__main__':
   if 'runpage' not in st.session_state :
      st.session_state.runpage = main

   st.session_state.runpage()
