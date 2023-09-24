from keras.layers import Input, SimpleRNN, LSTM, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer

import numpy as np
import re


with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')
    text = re.sub(r'[^А-я ]', '', text)

path = 'models/char/len10p_90.h5'
num_chars = 34  # 33 буквы + пробел
tokenizer = Tokenizer(num_words=num_chars, char_level=True)
tokenizer.fit_on_texts([text])

input_chars = 10
data = tokenizer.texts_to_matrix(text)
print(tokenizer.word_index)
print(data.shape)

# Входы по input_chars символов для предсказания
X = np.array(list(data[i:i+input_chars, :] for i in range(data.shape[0] - input_chars)))
Y = data[input_chars:]  # Символ который нужно предсказать

model = Sequential([
    Input(shape=(input_chars, num_chars)),
    Dense(64, activation='relu'),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(64, activation='tanh'),
    # SimpleRNN(128, activation='tanh'),
    Dense(num_chars, activation='softmax')])

model.compile(optimizer=Adam(clipnorm=1.), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, Y, batch_size=64, epochs=90)
model.save(path, save_format='h5')
# self_model = load_model(path)


def build_phrase(input_str, str_len=50):
    for i in range(str_len - len(input_str)):
        x = []
        for j in range(-input_chars, 0):  # range(i, i + input_chars):
            x.append(tokenizer.texts_to_matrix(input_str[j]))

        one_hot_x = np.array(x)
        inp = one_hot_x.reshape(1, input_chars, num_chars)

        pred = model.predict(inp, verbose=0)
        output_char = tokenizer.index_word[np.argmax(pred, axis=1)[0]]

        input_str += output_char

    return input_str


print(build_phrase('утренний с', str_len=120))
print(build_phrase('опыт не приобретай ', str_len=150))
print(build_phrase('человек не враг', str_len=150))
