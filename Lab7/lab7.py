import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
import re
import json
import swifter
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

print("Loading Yelp review data...")
yelp_data_path = r"/content/yelp_academic_dataset_review.json"
raw_reviews = []
with open(yelp_data_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        raw_reviews.append(json.loads(line))
        if i >= 50000:
            break
df = pd.DataFrame(raw_reviews)
print(f"Successfully loaded {len(df)} entries.")

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("Cleaning review content...")
df['clean_text'] = df['text'].swifter.apply(preprocess_text)
print("Text preprocessing complete.")

df['label'] = df['stars'].apply(lambda x: 1 if x > 3 else 0)
print("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

print("Converting text to sequences...")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

sequence_length = 100
X_train_pad = pad_sequences(train_sequences, maxlen=sequence_length, padding='post')
X_test_pad = pad_sequences(test_sequences, maxlen=sequence_length, padding='post')
y_train = np.array(y_train)
y_test = np.array(y_test)
print("Tokenization and padding completed.")

print("Constructing LSTM model...")
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=sequence_length),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Training the model...")
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

print("Evaluating on test data...")
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Set Accuracy: {accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def analyze_sentiment(input_text):
    cleaned = preprocess_text(input_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=sequence_length, padding='post')
    prob = model.predict(padded)[0][0]
    return "Positive" if prob > 0.5 else "Negative"

print("Testing samples from dataset:")
indices = np.random.randint(0, len(X_test), 2)
for i in indices:
    original = df.iloc[X_test.index[i]]['text']
    print(f"Review: {original}")
    print(f"Sentiment: {analyze_sentiment(original)}\n")

print("Testing samples not from dataset:")
sample_1 = "Best coffee I’ve ever had in my entire life. Will definitely come again!"
print(f"Review: {sample_1}\nSentiment: {analyze_sentiment(sample_1)}")
sample_2 = "The place was dirty, the food was cold, and the staff was rude."
print(f"Review: {sample_2}\nSentiment: {analyze_sentiment(sample_2)}")
