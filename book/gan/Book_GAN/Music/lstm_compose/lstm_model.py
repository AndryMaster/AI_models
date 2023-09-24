from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate, \
                         Reshape,  Flatten, RepeatVector, Permute, TimeDistributed, Multiply, Lambda
import keras.backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import RMSprop

import os
import pickle
import numpy as np

from music21 import chord, note
from loader import get_music_list, get_distinct, prepare_sequences, create_lookups

# data params
intervals = range(1)
seq_len = 32
data_folder = '../dataset/midi_songs_2'
build_data = False

if build_data:
    music_list, parser = get_music_list(data_folder)
    print(len(music_list), 'files in total')

    notes = []
    durations = []

    for idx, file in enumerate(music_list):
        print(f'{idx+1}: Parsing {file}', end='\t')

        try:
            original_score = parser.parse(file).chordify()
        except Exception as ex:
            print(f'skipped ({ex})')
            continue
        finally:
            print()

        for interval in intervals:
            notes.extend(['START'] * seq_len)
            durations.extend([0] * seq_len)

            for element in original_score.flat:
                if isinstance(element, chord.Chord):
                    notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                    durations.append(element.duration.quarterLength)
                if isinstance(element, note.Note):
                    if element.isRest:
                        notes.append(str(element.name))
                    else:
                        notes.append(str(element.nameWithOctave))
                    durations.append(element.duration.quarterLength)

    with open(os.path.join(data_folder, 'notes'), 'wb') as f:
        pickle.dump(notes, f)  # ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'B3', 'D3', 'G2',...]
    with open(os.path.join(data_folder, 'durations'), 'wb') as f:
        pickle.dump(durations, f)  # [0.25, 0.5, 0.25, 0.75, 0.25, Fraction(1, 3), 0.25,...]
else:
    with open(os.path.join(data_folder, 'notes'), 'rb') as f:
        notes = pickle.load(f)
    with open(os.path.join(data_folder, 'durations'), 'rb') as f:
        durations = pickle.load(f)

# print(notes)
# print(durations)

note_names, n_notes = get_distinct(notes)
duration_names, n_durations = get_distinct(durations)
distincts = [note_names, n_notes, duration_names, n_durations]
with open(os.path.join(data_folder, 'distincts'), 'wb') as f:
    pickle.dump(distincts, f)

note_to_int, int_to_note = create_lookups(note_names)
duration_to_int, int_to_duration = create_lookups(duration_names)
lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]
with open(os.path.join(data_folder, 'lookups'), 'wb') as f:
    pickle.dump(lookups, f)

network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len=seq_len)
# print(*note_to_int.items(), sep='\n')
# print(*duration_to_int.items(), sep='\n')
# print(n_notes, n_durations)

# print('note input')
# print(network_input[0][0])
# print('duration input')
# print(network_input[1][0])
# print('note output')
# print(network_output[0][0])
# print('duration output')
# print(network_output[1][0])

print(network_input[0].shape, network_input[1].shape)
print(n_notes, n_durations)


# model params
fit = False
rnn_units = 256  # 256
embed_size_note = 140  # in 470       _eq_
embed_size_duration = 70  # in 17

notes_in = Input(shape=(None,))  # None = seq_len
durations_in = Input(shape=(None,))

emb1 = Embedding(n_notes, embed_size_note)(notes_in)
emb2 = Embedding(n_durations, embed_size_duration)(durations_in)

x = Concatenate()([emb1, emb2])

x = LSTM(rnn_units, recurrent_dropout=0.2, return_sequences=True)(x)
x = LSTM(rnn_units, recurrent_dropout=0.2, return_sequences=True, dropout=0.15)(x)

e = Dense(1, activation='tanh')(x)
e = Reshape((-1,))(e)
alpha = Activation('softmax')(e)

alpha_repeat = Permute([2, 1])(RepeatVector(rnn_units)(alpha))  # Permute(shape2swap) [units, seq] -> [seq, units]

context = Multiply()([x, alpha_repeat])
context = Lambda(lambda arr: K.sum(arr, axis=1), output_shape=(rnn_units,))(context)

notes_out = Dense(n_notes, activation='softmax', name='notes')(context)
durations_out = Dense(n_durations, activation='softmax', name='durations')(context)


# attention model
if fit:
    model = Model([notes_in, durations_in], [notes_out, durations_out])
else:
    model = load_model('../models/compose_model_2mid32.h5', custom_objects={'K': K})
model.summary()
model.compile(optimizer=RMSprop(0.0016), metrics=['accuracy'],
              loss=['categorical_crossentropy', 'categorical_crossentropy'])

att_model = Model([notes_in, durations_in], alpha)

# fit
def schedule(epoch: int, lr: float):
    if epoch < 30:
        return np.linspace(0.002, 0.0004, 30)[epoch]
    return 0.0004

callbacks = [
    ModelCheckpoint('../models/compose_model_2mid.h5',
                    monitor='loss', mode='min',
                    save_best_only=True),
    EarlyStopping(monitor='loss', mode='min', patience=10),
    LearningRateScheduler(schedule)
]

if fit:
    model.fit(network_input, network_output,
              epochs=1000, batch_size=35,
              callbacks=callbacks)
