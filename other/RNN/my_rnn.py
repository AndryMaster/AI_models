import keras.callbacks
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, GRU, Dropout, Bidirectional,\
                         GlobalAveragePooling1D, MaxPooling1D, Conv1D, Dense
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def remove_URL(text):  # to remove URLs
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_html(text):  # to remove html tags
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def plot_history(history: keras.callbacks.History):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', c='green', lw='2')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', c='orangered', lw='2')
    plt.title('Accuracy', loc='left', fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss', c='green', lw='2')
    plt.plot(history.history['val_loss'], label='Validation Loss', c='orangered', lw='2')
    plt.title('Loss', loc='left', fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


data_frame = pd.read_csv('IMDB Dataset.csv')  # dataset.info()  print(dataset['sentiment'].value_counts())

data_frame['review'] = data_frame['review'].apply(remove_URL)
data_frame['review'] = data_frame['review'].apply(remove_html)

data = data_frame.to_numpy()
texts, labels = data[:, 0], data[:, 1]
print(data.shape, 'Loading texts...', sep='\n')

# texts = np.array(list(map(lambda y: y.replace('<br />', ''), texts)))
labels[labels=='positive'] = 0
labels[labels=='negative'] = 1
Y_one_hot = to_categorical(labels, num_classes=2)
# print(Y[:15], Y_one_hot[:15], texts[:5])

max_words_count = 750  # 8000
tokenizer = Tokenizer(num_words=max_words_count, lower=True, split=' ', char_level=False,
                      filters='!–"—#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»')
tokenizer.fit_on_texts(texts)
# all_words_ = list(tokenizer.word_counts.items())
# print(f'All words: {len(all_words_)}\nTop 30 words: {sorted(all_words_, reverse=True, key=lambda a: a[1])[:30]}')

max_text_len = 500  # median
X = tokenizer.texts_to_sequences(texts)
X_pad = pad_sequences(X, maxlen=max_text_len)
# print(*X[:5], sep='\n')
# print(*X_pad[:5], sep='\n')

count = np.zeros(len(X))
for i in range(len(X)):
    count[i] = len(X[i])
print(np.min(count), np.max(count))
print(np.mean(count), np.median(count))

embedding_dim = 32
X_train, X_test = X_pad[:40_000], X_pad[-10_000:]
Y_train, Y_test = Y_one_hot[:40_000], Y_one_hot[-10_000:]
del X, X_pad, count, Y_one_hot, texts, labels, data

path = f'models/my_imdb/v1_{max_words_count}_{embedding_dim}_{max_text_len}_gru.h5'
model_1 = Sequential([
    Embedding(max_words_count, embedding_dim, input_length=max_text_len),
    GRU(128, return_sequences=True),
    GRU(96),
    Dense(64, activation='sigmoid'),
    Dense(2, activation='softmax')])
model_2 = Sequential([
    Embedding(max_words_count, embedding_dim, input_length=max_text_len),
    Conv1D(128, 5, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')])
model_3 = Sequential([
    Embedding(max_words_count, embedding_dim, input_length=max_text_len),
    Bidirectional(GRU(128, return_sequences=True)),
    Bidirectional(GRU(128)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')])
model_4 = Sequential([  # 80,506
    Embedding(max_words_count, embedding_dim, input_length=max_text_len),
    Conv1D(32, 3, padding='same', activation='relu'),
    Conv1D(32, 3, padding='same', activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(100),
    Dense(2, activation='softmax')])
model_5 = Sequential([  # 77,402
    Embedding(max_words_count, embedding_dim, input_length=max_text_len),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(2, activation='softmax')])

model = model_4
model.compile(optimizer=Adam(0.0002, clipnorm=1.), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=5)
# model.save(path, save_format='h5')
# model = load_model(path)
plot_history(hist)


def smart_predict(count_predicts=3, is_test=True):
    indexes = np.random.random_integers(0, X_test.shape[0], count_predicts)
    for i in indexes:
        data2pred, label = (X_test[i], np.argmax(Y_test[i])) if is_test else (X_train[i], np.argmax(Y_train[i]))
        data2pred = np.expand_dims(data2pred, 0)
        result = model.predict(data2pred, verbose=0, batch_size=1)

        data2word = data2pred[data2pred!=0]
        words = [reverse_word_map.get(j) for j in data2word]
        print(f'Id : {i}\t{result}\t{label==np.argmax(result)}\t'
              f'{["positive", "negative"][np.argmax(result)]}\n{" ".join(words)}\n')


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
# smart_predict(count_predicts=2, is_test=False)
smart_predict(count_predicts=10)
