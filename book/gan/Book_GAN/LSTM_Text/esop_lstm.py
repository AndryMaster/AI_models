from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dropout, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop

from keras.preprocessing.text import Tokenizer
from keras import utils

import numpy as np
import re

filename = 'datasets/esop_data.txt'
filename_clear = 'datasets/esop_clear.txt'
seq_length = 25
cleared = True
start_story = '| ' * seq_length

if not cleared:
    with open(filename, 'r', encoding='utf-8-sig') as file:
        text = file.read()

        # ОЧИСТКА
        text = text.lower()
        text = text.replace('\n' * 5, start_story)
        text = text.replace('\n', ' ')
        text = text.replace('"', ' " ')
        text = re.sub('([!»#$ %&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        # text = re.sub(' +', '. ', text).strip()
        # text = text.replace('..', '.')

    with open(filename_clear, 'w', encoding='utf-8-sig') as file:
        file.write(text)
else:
    with open(filename_clear, 'r', encoding='utf-8-sig') as file:
        text = file.read()


def generate_sequences(token_list_, step):
    x_ = []
    y_ = []
    for i in range(0, len(token_list_) - seq_length, step):
        x_.append(token_list_[i: i + seq_length])
        y_.append(token_list_[i + seq_length])
    y_ = utils.to_categorical(y_, num_classes=total_words)
    return np.array(x_), np.array(y_), len(x_)


tokenizer = Tokenizer(lower=True, char_level=False, filters='')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
token_list = tokenizer.texts_to_sequences([text])[0]

x, y, num_seq = generate_sequences(token_list, step=1)
print(f'Number of words: {total_words}')
print(f'Number of sequences: {num_seq}')
# print(token_list[:120])
print(x.shape, y.shape)
train = False
model_file = 'models/esop_2g.h5'

if train:
    r_units = 256
    embedding_size = 100

    model = Sequential([
        Input(shape=(seq_length,)),
        Embedding(total_words, embedding_size),
        GRU(r_units, return_sequences=True),
        GRU(r_units),
        Dropout(0.15),
        Dense(total_words, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer=RMSprop(0.0007),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y,  # validation_split=0.03,
              epochs=100, batch_size=35, shuffle=True)
    model.save(model_file)
else:
    model = load_model(model_file)
    model.summary()

# Test
def sample_with_temp(input_pred, temperature=1.0):  # 0 = model, 1 = random; (0, 1]
    pred = np.array(input_pred).astype('float64')
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    probas = np.random.multinomial(1, pred, 1)
    return np.argmax(probas)


def generate_text(seed_text, next_words, max_sequence_len, temp):
    output_text = seed_text

    seed_text = start_story + seed_text

    for _ in range(next_words):
        token_list_ = tokenizer.texts_to_sequences([seed_text])[0]
        token_list_ = token_list_[-max_sequence_len:]
        token_list_ = np.reshape(token_list_, (1, max_sequence_len))

        pred = model.predict(token_list_, verbose=0)[0]
        y_class = sample_with_temp(pred, temperature=temp)

        if y_class == 0:
            output_word = ''
        else:
            output_word = tokenizer.index_word[y_class]

        output_text += output_word + ' '
        seed_text += output_word + ' '

        if output_word == "|":
            break

    return output_text


def human_led_text(text_to_continue):
    print(text_to_continue + ' ...')
    seed_text = start_story + text_to_continue

    token_list_ = tokenizer.texts_to_sequences([seed_text])[0]
    token_list_ = token_list_[-seq_length:]
    token_list_ = np.reshape(token_list_, (1, seq_length))

    probs = model.predict(token_list_, verbose=0)[0]

    top_10_idx = np.flip(np.argsort(probs)[-10:])
    top_10_probs = [probs[x] for x in top_10_idx]
    top_10_words = tokenizer.sequences_to_texts([[x] for x in top_10_idx])

    for i, (prob, word) in enumerate(zip(top_10_probs, top_10_words)):
        print(f"{i+1}: {prob*100:.1f}% - {word}")

    chosen_word = int(input('Choose word idx: '))

    if 0 <= chosen_word <= 9:
        return top_10_words[chosen_word-1]
    else:
        return None


new_text = "the frog and the snake . "
gen_words = 450
print('Temp 0.05')
print(generate_text(new_text, gen_words, seq_length, temp=0.05), end='\n' * 5)
print('Temp 0.1')
print(generate_text(new_text, gen_words, seq_length, temp=0.1), end='\n' * 5)
print('Temp 0.25')
print(generate_text(new_text, gen_words, seq_length, temp=0.25), end='\n' * 5)
print('Temp 0.5')
print(generate_text(new_text, gen_words, seq_length, temp=0.5), end='\n' * 5)
print('Temp 1.0')
print(generate_text(new_text, gen_words, seq_length, temp=1), end='\n' * 5)

print(human_led_text('i want to go'))
print(human_led_text('hi , my dear'))
print(human_led_text('the lion said ,'))
print(human_led_text('the lion said , and'))
print(human_led_text('the cat was seen the mice , it think that , the mice may be'))
