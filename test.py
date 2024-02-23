import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load and preprocess data
data = pd.read_csv('spotify_dataset.csv')
data['lyrics'] = data['lyrics'].str.lower()
stop_words = set(stopwords.words('english'))
data['lyrics'] = data['lyrics'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words and word not in string.punctuation]))

# Tokenize and pad sequences
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['lyrics'].values)
X = tokenizer.texts_to_sequences(data['lyrics'].values)
X = pad_sequences(X, maxlen=max_len)

# Train-Test Split
Y = pd.get_dummies(data['song_name']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build LSTM Model
embedding_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
batch_size = 32
model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2)

# Model Testing
test_text = "the sun is shining bright"
test_text = tokenizer.texts_to_sequences([test_text])
test_text = pad_sequences(test_text, maxlen=max_len)
prediction = model.predict(test_text)
predicted_song_index = np.argmax(prediction)
predicted_song_name = tokenizer.index_word[predicted_song_index]
print(f"The predicted song is: {predicted_song_name}")
