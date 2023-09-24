from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import Adam

import numpy as np


with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')
    # text_of_words = re.sub(r'[^А-я ]', '', text_of_words)

maxWordsCount = 1000  # 10_000 = human, 20_000 = good
tokenizer = Tokenizer(num_words=maxWordsCount, lower=True, split=' ', char_level=False,
                      filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»')
tokenizer.fit_on_texts([text])
print(tokenizer.word_counts.items())

input_words = 4
data = np.array(tokenizer.texts_to_sequences([text])[0])

X = np.array([data[i:i + input_words] for i in range(len(data) - input_words)])
Y = to_categorical(data[input_words:], num_classes=maxWordsCount)
print(data.shape, X.shape, Y.shape, sep='\n')

path = 'models/word/v5_4.h5'
model = Sequential([
    Embedding(maxWordsCount, 256, input_length=input_words),
    SimpleRNN(128, activation='tanh', return_sequences=True),
    SimpleRNN(128, activation='tanh'),
    Dense(maxWordsCount, activation='softmax')])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, Y, batch_size=50, epochs=60)
model.save(path, save_format='h5')
# self_model = load_model(path)


def build_phrase(text_of_words, count_words=26):
    res = text_of_words.split()
    data_pharse = tokenizer.texts_to_sequences([text_of_words])[0]  # list()
    for i in range(count_words):
        # x = to_categorical(dataset[i: i + inp_words], num_classes=max_words_count)  преобразуем в One-Hot-encoding
        # преобразуем в One-Hot-encoding
        # emb = x.reshape(1, inp_words, max_words_count)
        x = data_pharse[i: i + input_words]  # -input_words:
        inp = np.expand_dims(x, axis=0)

        pred = model.predict(inp, verbose=0)
        index_word = np.argmax(pred, axis=1)[0]
        data_pharse.append(index_word)
        res.append(tokenizer.index_word[index_word])

    return ' '.join(res)


print(build_phrase('позитив добавляет годы а'))
print(build_phrase('кому ты нужен некрасивый'))
print(build_phrase('все новое опасно если'))
