import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string
def remove_punctuation(input_string):
    translation_table = str.maketrans('', '', string.punctuation)
    result = input_string.translate(translation_table)
    return result
import re
def remove_words_with_apostrophes(input_string):
    pattern = r'\b\w+\'\w+\b'
    result = re.sub(pattern, '', input_string)
    return result

df=pd.read_csv("spliced_spotify_dataset.csv")
#df_real = pd.read_csv("Spotify Million Song Dataset_exported.csv")

# Vectorize the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Input snippet
input_snippet = input("Enter a snippet to find the song's name and it's artist : ")
input_snippet=remove_words_with_apostrophes(input_snippet)
input_snippet=remove_punctuation(input_snippet)

# Vectorize input snippet
input_snippet_vector = vectorizer.transform([input_snippet])

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(X, input_snippet_vector)

# Identify the unique song
unique_song_index = similarity_scores.argmax()
unique_song = df.iloc[unique_song_index]

print("..........Finding..........\n")

print("<-- Result found --> \n")
print(f'Song: {unique_song["song"]}\n')
print(f'Artist: {unique_song["artist"]}\n')
#print(f'Lyrics: {unique_song["text"]}')