import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv("Spotify Million Song Dataset_exported.csv")

# Preprocess the text
indent = ["\n", "\r", "\t"]
def remove_indents(text, indent):
    for elements in indent:
        text = text.replace(elements, '')
    return text

df['Lyrics'] = df['Lyrics'].str.lower()
df['Lyrics'] = df['Lyrics'].apply(lambda elements: remove_indents(elements, indent))

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Lyrics'])
sequences = tokenizer.texts_to_sequences(df['Lyrics'])

# Pad the sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Prepare the labels
label_encoder = LabelEncoder()
df['artist_encoded'] = label_encoder.fit_transform(df['Artist'])
df['song_encoded'] = label_encoder.fit_transform(df['Song'])

# Train-test split
X_train, X_test, y_train_artist, y_test_artist, y_train_song, y_test_song = train_test_split(
    padded_sequences, df['artist_encoded'], df['song_encoded'], test_size=0.2, random_state=42)

# Define the model
embedding_dim = 300
lstm_out = 300
units = 100

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=units))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax')) # Adjust units based on the number of classes

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_artist, epochs=25, batch_size=64, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_artist)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
