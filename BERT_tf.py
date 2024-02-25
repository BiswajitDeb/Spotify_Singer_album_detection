import pandas as pd
import numpy as np
import re
import string
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
df = pd.read_csv("/content/Spotify Million Song Dataset_exported.csv", error_bad_lines=False, engine="python")

# Drop unnecessary columns
df = df.drop(['link'], axis=1)

# Clean the text data
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = text.replace('\n', ' ')  # Remove newline characters
    text = text.replace('\r', ' ')  # Remove carriage return characters
    text = text.replace('\t', ' ')  # Remove tab characters
    text = re.sub(r'\b\w+\'\w+\b', '', text)  # Remove words with apostrophes
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

df['text'] = df['text'].apply(clean_text)

# Concatenate artist and song names
df['combined'] = df['artist'] + ' ' + df['song'] + ' ' + df['text']

# Tokenize the combined text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded_inputs = tokenizer(df['combined'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='tf')

# Encode combined labels
label_encoder = LabelEncoder()
df['combined_encoded'] = label_encoder.fit_transform(df['combined'])

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(encoded_inputs['input_ids'], df['combined_encoded'], test_size=0.2, random_state=42)

# Create a TensorFlow Dataset for training and testing sets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

# Batch and shuffle the datasets
train_dataset = train_dataset.shuffle(buffer_size=100).batch(8)
test_dataset = test_dataset.batch(8)

# Load the pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train the model
model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Make predictions on test set
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions.logits, axis=1)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predicted_labels)
true_labels = label_encoder.inverse_transform(test_labels.numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy}')

# Display classification report
print(classification_report(true_labels, predicted_labels))
