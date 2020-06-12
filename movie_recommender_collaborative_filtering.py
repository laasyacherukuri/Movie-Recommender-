# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:45:58 2020

@author: Laasya Cherukuri
"""
#We are using item to item collaborative filtering
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("toy_dataset.csv",index_col = 0)
ratings = ratings.fillna(0)
#print(ratings.head())

def standardise(row):
    new_row = (row - row.mean())/(row.max() - row.min())
    return new_row

rating_std = ratings.apply(standardise)
#print(rating_std)

item_similarity = cosine_similarity(rating_std.T)
#print(item_similarity)

#Creating dataframe

item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns)

def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity_df[movie_name]*user_rating
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score
#print(get_similar_movies("action1",5))

action_lover = [("action1",5),("romantic2",1),("romantic3",1)]
similar_movies=pd.DataFrame()
for movie,rating in action_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)
print(similar_movies.sum().sort_values(ascending=False))