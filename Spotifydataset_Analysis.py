#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df_tracks= pd.read_csv("E:/My projects/Python/tracks.csv")
df_tracks.head() 
# // used to get first n rows //


# In[7]:


#check null values with sum function that gives null values in each column
pd.isnull(df_tracks).sum()


# In[8]:


#check info about datset, what are number of rows and columns, memory usage
df_tracks.info()


# In[9]:


#10 least popular song present on spotify
sorted_df = df_tracks.sort_values("popularity", ascending = "True").head(10)
sorted_df


# In[10]:


#get descriptive statistics for numerical columns present in the dataset
# transpose() Reflect the DataFrame over its main diagonal by writing rows as columns and vice-versa.
df_tracks.describe().transpose()


# In[11]:


#tracks that have popularity > 90
#query() in pandas, Query the columns of a DataFrame with a boolean expression.
most_popular = df_tracks.query("popularity>90", inplace=False).sort_values("popularity",ascending=False)
most_popular[:10]


# In[12]:


#set index to be release date column 
#set_index() Set the DataFrame index using existing columns.
df_tracks.set_index("release_date",inplace= True)

#Immutable sequence used for indexing and alignment. 
#This function converts a scalar, array-like, Series or DataFrame/dict-like to a pandas datetime object.
#converting the set index to pandas datetime format
df_tracks.index=pd.to_datetime(df_tracks.index)
df_tracks.head()


# In[13]:


#check the artist in 18th row in the dataset
df_tracks[["artists"]].iloc[18]


# In[14]:


#check the song name at 18th row and the artist
df_tracks[["name","artists"]].iloc[18]


# In[15]:


#convert the song duration from miliseconds to just seconds using apply and lambda function
#Apply a function along an axis of the DataFrame, lambda function to both the columns and rows of the Pandas data frame.
#axis=1 applies the function along the rows, inplace = True  changes the default behaviour such that the operation on the dataframe doesn't return anything,
#it instead 'modifies the underlying data' (more on that later).

df_tracks["duration1"]=df_tracks["duration_ms"].apply(lambda x : round(x/1000))
df_tracks.drop("duration_ms", inplace=True, axis=1)


# In[16]:


df_tracks.head()


# In[17]:


df_tracks


# In[18]:



corr_df= df_tracks.drop(["key","explicit","mode"],axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))

#vmax,vmin Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
#annot If True, write the data value in each cell. #cmap The mapping from data values to color space

heatmap=sns.heatmap(corr_df,annot=True, fmt= ".1g", vmin=-1, vmax=1, center=0,cmap="inferno", linewidths=1,linecolor = "Black" )
heatmap.set_title("Correlations heatmap between variables")
heatmap.set_xsticklabels(heatmap.get_xsticklabels(), rotation = 90)


# In[ ]:


#take 0.4 percent of this data and create two regression plot our of it
# create a sample dataframe using sample() Return a random sample of items from an axis of object.
sample_df= df_tracks.sample(int(0.004 * len(df_tracks)))


# In[ ]:


print(len(sample_df))


# In[ ]:


#create a regression plot between loudness and energy
plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y="loudness", x="energy", color = "c").set(title="Loudness vs Energy Correlation")


# In[ ]:





# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df, y="popularity", x="acousticness", color = "b").set(title="Loudness vs Energy Correlation")


# In[20]:


#Return an Index of values for requested level.
#This is primarily useful to get an individual level of values from a MultiIndex, but is provided on Index as well for compatibility.
df_tracks["dates"]= df_tracks.index.get_level_values("release_date")
df_tracks.dates=pd.to_datetime(df_tracks.dates)
years=df_tracks.dates.dt.year


# In[ ]:


pip install --user seaborn==0.11.0


# In[23]:


#create a distribution plot to visualize the total number of songs in each year since 1922 that is available on spotify
sns.displot(years,discrete= True, aspect = 2, height = 5, kind = "hist").set(title="Number of songs per year")


# In[28]:


total_dr = df_tracks.duration1
fig_dims= (18,8)
fig,ax = plt.subplots(figsize= fig_dims)
fig = sns.barplot(x=years, y= total_dr, errwidth = False).set(title="years vs duration")
plt.xticks(rotation=90)


# In[30]:


total_dr = df_tracks.duration1
sns.set_style(style="whitegrid")
fig_dims= (10,5)
fig,ax = plt.subplots(figsize= fig_dims)
fig = sns.lineplot(x=years, y= total_dr, ax = ax).set(title="years vs duration")
plt.xticks(rotation=60)


# In[31]:


df_genre=pd.read_csv("E:/My projects/Python/SpotifyFeatures.csv")


# In[37]:


df_genre.head(5)


# In[40]:


#duration of songs for different genre
plt.title("Duration of the songs in different genre")
sns.color_palette("rocket", as_cmap= True)
sns.barplot(y= "genre", x= "duration_ms", data = df_genre )
plt.xlabel("Duration in milliseconds")
plt.ylabel("genre")


# In[42]:


#top most 5 genre on basis of popularity
sns.set_style(style="darkgrid")
plt.figure(figsize = (10,5))
famous=df_genre.sort_values("popularity",ascending=False).head(10)
sns.barplot(y="genre",x="popularity",data=famous).set(title="Top 5 genre by popularity")


# In[ ]:




