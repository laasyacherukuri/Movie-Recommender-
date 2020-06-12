# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:20:43 2020

@author: Laasya Cherukuri
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London London Paris", "Paris Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
print(count_matrix.toarray())

similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)