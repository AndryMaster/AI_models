from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import numpy as np


with open('train_text_true.txt', 'r', encoding='utf-8') as f:
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '')

with open('train_text_false.txt', 'r', encoding='utf-8') as f:
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '')


texts = texts_true + texts_false
count_true = len(texts_true)
count_false = len(texts_false)
total_lines = count_true + count_false
print(count_true, count_false, total_lines)

max_words_count = 1000
tokenizer = Tokenizer(num_words=max_words_count, lower=True, split=' ', char_level=False,
                      filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»')
tokenizer.fit_on_texts(texts)
print(list(tokenizer.word_counts.items()))

max_text_len = 10
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad)

X = data_pad
Y = np.array([[1, 0]] * count_true + [[0, 1]] * count_false)
print(X.shape, Y.shape)

indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X, Y = X[indices], Y[indices]

path = 'models/gru/phrase_select.h5'
model = Sequential([
    Embedding(max_words_count, 128, input_length=max_text_len),
    GRU(128, return_sequences=True),
    GRU(64),
    Dense(2, activation='softmax')])

model.compile(optimizer=Adam(0.0005, clipnorm=1.), loss='categorical_crossentropy', metrics=['accuracy'])  # clipnorm
model.summary()

model.fit(X, Y, batch_size=32, epochs=128)
# model.save(path, save_format='h5')
# model = load_model(path)


def smart_predict(test_text):
    test_data = tokenizer.texts_to_sequences([test_text])
    test_data_pad = pad_sequences(test_data, maxlen=max_text_len)
    result = model.predict(test_data_pad, verbose=0)

    words = [reverse_word_map.get(i) for i in test_data[0]]
    print(f'{" ".join(words)}\n{result}\t{np.argmax(result)}')


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
smart_predict('не доверяй никому')
smart_predict('я притягиваю только плохое')
smart_predict('позитивный настрой это потенциал к успеху')
smart_predict('все будет хорошо если увидите в людях добро')  # и плохое
