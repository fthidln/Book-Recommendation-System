# -*- coding: utf-8 -*-
"""Submission_Recommender System_Muhammad Fatih Idlan

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QVEZmzCuHntVOppUesRWuLAPE1xFkFg6

# **Recommendation System: Automating Book's Suggestion using Collaborative Filtering**

By    : Muhammad Fatih Idlan (faiti.alfaqar@gmail.com)

## Project Overview
In today’s digital age, the volume of content and choices available to users across platforms is overwhelming. Recommender systems play an indispensable role in navigating this vast landscape, ensuring users discover relevant and engaging content without being inundated by irrelevant options. By personalizing user experiences, these systems have become a cornerstone in industries like e-commerce, entertainment, and education, boosting user satisfaction, retention, and revenue. This project delves into the development of a book recommendation system, leveraging collaborative filtering techniques and the Nearest Neighbors algorithm to match users with books they are most likely to enjoy. Collaborative filtering, a widely used approach, relies on user-item interactions to uncover patterns and provide recommendations. Furthermore, Collaborative Filtering can make unforeseen recommendations, which means it might offer items that are relevant to the user even if the information is not in the user's profile [[ 1 ]](https://doi.org/10.1016/j.eij.2015.06.005). The system implementation is aimed to demonstrate how machine learning can be harnessed to create a seamless and personalized user experience in the context of literature discovery.

## Business Understanding
### Problem Statement
Starting with explanation from the background above, core problems that this project aims to solve are:

* How to develop a machine learning-based recommendation system for books?
* How are the results when data with and without standardization is compared using the same algorithm?

### Objectives
According to problem statement above, this project has several objectives too, that are:

* Develop a machine learning-based recommendation system for books
* Determining high performance model with variation of data preparation method

### Solution Approach
To achive the objectives, we need to perform several things such as:

* Using Nearest Neighbour through variation of data with and without standardization to selecting high performance corresponding to evaluation metrics (Euclidean Distance)

## Import Package dan Libraries
"""

import kagglehub
import kagglehub
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

"""## Data Loading"""

path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")

print("Path to dataset files:", path)

files = os.listdir(path)

print("Files in dataset:", files)

path2book = f'{path}/Books.csv'
book = pd.read_csv(path2book)
book.head()

path2rating = f'{path}/Ratings.csv'
rating = pd.read_csv(path2rating)
rating.head()

book = book.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'Year-Of-Publication', 'Publisher'], axis=1)

book.info()

rating.info()

"""## Data Cleaning"""

book.isna().sum()

book = book.dropna()

book.isna().sum()

rating.isna().sum()

print(f'Duplicated data: {book.duplicated().sum()}')

print(f'Duplicated data: {rating.duplicated().sum()}')

"""## Data Understanding

The dataset that used in this project is Book Recommendation Dataset, which can be accessed through kaggle [[ 2 ]](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data). This dataset consist of 3 csv files, Books.csv (271360 rows with 8 columns), Ratings.csv (1149780 rows with 3 columns), and Users.csv (27885 rows with 3 columns), also has 3 png file which irrelevant in this project. The explanation for each column can be seen below:

For Books.csv, the column are consist of:

* ISBN = International Standard Book Number of the books inside obtained from Amazon Web Services
* Book-Title = Title of the books obtained from Amazon Web Services
* Book-Author = The Author of the books obtained from Amazon Web Services
* Year-Of-Publication = Publication year of the books obtained from Amazon Web Services
* Publisher = The Publisher of the books obtained from Amazon Web Services
* Image-URL-S = URL for small sized Book's cover images point to the Amazon web site
* Image-URL-M = URL for medium sized Book's cover images point to the Amazon web site
* Image-URL-L = URL for large sized Book's cover images point to the Amazon web site

For Ratings.csv, the column are consist of:

* User-ID = Anonymized user identification in integers
* ISBN = International Standard Book Number of the books inside obtained from Amazon Web Services
* Book-Rating = Rating of the books, expressed on a scale from 1-10 (higher values denoting higher appreciation) in an explicit way, or expressed by 0 in implicit way

For Users.csv, the column are consist of:

* User-ID = Anonymized user identification in integers
* Location = Region of the reader in form of city, country
* Age = Age of the readers
"""

title = book['Book-Title'].value_counts()

print(f'Amount of unique book titles: {len(title)}')

isbn = book['ISBN'].value_counts()

print(f'Amount of unique ISBN: {len(isbn)}')

author = book['Book-Author'].value_counts()

print(f'Amount of unique author: {len(author)}')

user = rating['User-ID'].value_counts()

print(f'Amount of unique userID: {len(user)}')

"""### Exploratory Data Analysis"""

rating['Book-Rating'].hist(figsize=(5,5), bins=50)
plt.title('Distribution of Book Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

rating = rating[rating['Book-Rating'] != 0]

rating['Book-Rating'].hist(figsize=(5,5), bins=50)
plt.title('Distribution of Book Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

title_counts = book['Book-Title'].value_counts()

top_10_books = title_counts.head(10)

plt.figure(figsize=(10, 6))
plt.bar(top_10_books.index, top_10_books.values)
plt.xlabel("Book Title")
plt.ylabel("Amount")
plt.title("Top 10 Books by Amount")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

ratings = rating['User-ID'].value_counts()

ratings.sort_values(ascending=False).to_frame(name='Rating amount').head(10).plot(kind='bar',title='Top 10 Users with the Most Rating Amount')

top_10_books

rating.head()

book.head()

"""## Pre-Processing"""

len(ratings[ratings < 200])

rating['User-ID'].isin(ratings[ratings < 200].index).sum()

df_rating = rating[
  ~rating['User-ID'].isin(ratings[ratings < 200].index)
]
df_rating.shape

ratings = rating['ISBN'].value_counts()
ratings.sort_values(ascending=False).head()

len(ratings[ratings < 100])

book['ISBN'].isin(ratings[ratings < 100].index).sum()

df_rating = df_rating[
  ~df_rating['ISBN'].isin(ratings[ratings < 100].index)
]
df_rating.shape

books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
        "I'll Be Seeing You",
        "The Weight of Water",
        "The Surgeon",
        "I Know This Much Is True"]

for i in books:
    print(df_rating['ISBN'].isin(book[book['Book-Title'] == i]['ISBN']).sum())

df_main = df_rating.pivot_table(index=['User-ID'],columns=['ISBN'],values='Book-Rating').fillna(0).T
df_rating.info()

df_main.index = df_main.join(book.set_index('ISBN'))['Book-Title']

df_main

df_main.loc["The Divine Secrets of the Ya-Ya Sisterhood: A Novel"][:5]

"""##Model Development"""

KNN = NearestNeighbors(metric='euclidean')
KNN.fit(df_main.values)

df_main.iloc[0].shape

title = 'The Bean Trees'
df_main.loc[title].shape

distance_KNN, indice_KNN = KNN.kneighbors([df_main.loc[title].values], n_neighbors=10)

print(distance_KNN)
print(indice_KNN)

df_main.iloc[indice_KNN[0]].index.values

standarize_data = StandardScaler().fit_transform(df_main.values)

BT = NearestNeighbors(metric='euclidean')
BT.fit(standarize_data)

distance_BT, indice_BT = BT.kneighbors([df_main.loc[title].values], n_neighbors=10)

print(distance_BT)
print(indice_BT)

df_main.iloc[indice_BT[0]].index.values

"""## Model Evaluation"""

df_eval = pd.DataFrame({
    'Recommended Title w/ Standardization'   : df_main.iloc[indice_BT[0]].index.values,
    'Recommended Distance w/ Standardization': distance_BT[0],
    'Recommended Title w/o Standardization'   : df_main.iloc[indice_KNN[0]].index.values,
    'Recommended Distance w/o Standardization': distance_KNN[0],
    }) \
    .sort_values(by='Recommended Distance w/ Standardization', ascending=True)

df_eval

df_eval[['Recommended Distance w/ Standardization','Recommended Distance w/o Standardization']].plot(kind='line', figsize=(8, 4), title='Recommended Distance Comparison')
plt.gca().spines[['top', 'right']].set_visible(False)

"""## Top-N Recommendation Function"""

def get_recommends(title="", n=5):
    try:
        book_index = df_main.index.get_loc(title)  # Get index directly
        book_values = df_main.values[book_index]  # Get values using index
    except KeyError as e:
        print('The given book', e, 'does not exist')
        return

    n+=1

    distances, indices = KNN.kneighbors([book_values], n_neighbors=n)

    # Get recommended titles directly using indices
    recommended_titles = df_main.iloc[indices[0]].index.tolist()

    recommended_titles = [x for x in recommended_titles if x != title]

    recommended_titles_str = '\n'.join(recommended_titles)

    return f"This are your top {n-1} recommended books:\n{recommended_titles_str}"

books = get_recommends("Into the Wild",5)
print(books)