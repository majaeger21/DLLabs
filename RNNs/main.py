'''
In this lab, you will explore the capabilities of Recurrent Neural Networks (RNNs) through 
hands-on coding exercises. You will implement and train basic RNNs to solve two different tasks, 
gaining insights into their strengths and limitations.

Character Prediction

Task: Build an RNN to predict the next character in a given sequence of text.

Dataset: We will use a small Shakespeare text dataset provided with the code. (Tiny ShakespeareLinks to an external site.)

Libraries: This lab uses TensorFlow. Example code snippets are provided below.

Steps:
  Preprocess the text: Convert the text into numerical representation using one-hot encoding or character embedding.
  Build the RNN model: Implement a simple RNN cell (e.g., LSTM) and stack it to create a multi-layer RNN.
  Train the model: Train the RNN to predict the next character in a sequence, minimizing the loss function (e.g., cross-entropy).
  Evaluate the model: Generate text samples using the trained model and evaluate the quality based on coherence and fluency.
'''
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import requests
import os
import numpy as np

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data = requests.get(url).text

chars = sorted(set(data))
char_to_num = {char: idx for idx, char in enumerate(chars)}
num_to_char = {idx: char for idx, char in enumerate(chars)}

encoded_data = np.array([char_to_num[char] for char in data])

sequence_length = 50
sequence_step = 2
sequences = []
next_chars = []

for i in range(0, len(encoded_data) - sequence_length, sequence_step):
    sequences.append(encoded_data[i:i + sequence_length])
    next_chars.append(encoded_data[i + sequence_length])

X = np.array(sequences)
y = to_categorical(next_chars, num_classes=len(chars))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

vocab_size = len(chars)
embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

if os.path.exists('model.keras'):
    model = load_model('model.keras')
else:
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=128)
    model.save('model.keras')

def pick_char(probabilities, diversity=1.0):
    probabilities = np.asarray(probabilities).astype('float64')
    probabilities = np.power(probabilities, 1 / diversity)
    probabilities /= np.sum(probabilities)
    return np.argmax(np.random.multinomial(1, probabilities, 1))

def compose_text(length, diversity=1.0):
    generated_sequence = data[np.random.randint(0, len(data) - sequence_length - 1) :][:sequence_length]
    print(f'Initial seed: "{generated_sequence}"')

    for _ in range(length):
        encoded_sequence = np.array([[char_to_num[char] for char in generated_sequence]])
        next_character = num_to_char[pick_char(model.predict(encoded_sequence, verbose=0)[0], diversity)]
        generated_sequence = generated_sequence[1:] + next_character
        print(next_character, end='', flush=True)
    print()

compose_text(length=500, diversity=0.5)