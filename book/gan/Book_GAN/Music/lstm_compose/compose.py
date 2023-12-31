import os
import pickle
import numpy as np

from keras.models import load_model
from keras import backend as K

from music21 import chord, note, stream, instrument, duration
from loader import sample_with_temp

data_folder = '../dataset/midi_songs_2'

# prediction params
notes_temp = 1
duration_temp = 1
max_len_generated_notes = 136
max_seq_len = 64
seq_len = 64
midi_instrument = instrument.Piano()
output_filename = os.path.join('../output', 'out5.mid')

# notes = ['START', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3', 'D3', 'D3', 'E3', 'D3', 'G3', 'F#3']
# durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

# notes = ['START', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3','F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3', 'F#3', 'G#3', 'F#3', 'E3']
# durations = [0, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2, 0.75, 0.25, 1, 1, 1, 2]

notes = ['START', 'D3', 'E4', 'D3', 'A2', 'D5']
durations = [0, 0.75, 0.25, 0.75, 2, 0.75]

# notes = ['START']
# durations = [0]

# model and predict
model = load_model('../models/compose_model_2mid64.h5', custom_objects={'K': K})
model.summary()

with open(os.path.join(data_folder, 'distincts'), 'rb') as filepath:
    distincts = pickle.load(filepath)
    note_names, n_notes, duration_names, n_durations = distincts

with open(os.path.join(data_folder, 'lookups'), 'rb') as filepath:
    lookups = pickle.load(filepath)
    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups

if seq_len is not None:
    notes = ['START'] * (seq_len - len(notes)) + notes
    durations = [0] * (seq_len - len(durations)) + durations

sequence_length = len(notes)
prediction_output = []
notes_input_sequence = []
durations_input_sequence = []

overall_preds = []

for n, d in zip(notes, durations):
    note_int = note_to_int[n]
    duration_int = duration_to_int[d]

    notes_input_sequence.append(note_int)
    durations_input_sequence.append(duration_int)

    prediction_output.append([n, d])

    if n != 'START':
        midi_note = note.Note(n)

        new_note = np.zeros(128)
        new_note[midi_note.pitch.midi] = 1
        overall_preds.append(new_note)


for note_index in range(max_len_generated_notes):
    prediction_input = [
        np.array([notes_input_sequence]),
        np.array([durations_input_sequence])]

    notes_prediction, durations_prediction = model.predict(prediction_input, verbose=0)
    new_note = np.zeros(128)

    for idx, n_i in enumerate(notes_prediction[0]):
        try:
            note_name = int_to_note[idx]
            midi_note = note.Note(note_name)
            new_note[midi_note.pitch.midi] = n_i
        except:
            pass

    overall_preds.append(new_note)

    int_note = sample_with_temp(notes_prediction[0], notes_temp)
    int_duration = sample_with_temp(durations_prediction[0], duration_temp)

    note_result = int_to_note[int_note]
    duration_result = int_to_duration[int_duration]

    prediction_output.append([note_result, duration_result])

    notes_input_sequence.append(int_note)
    durations_input_sequence.append(int_duration)

    if len(notes_input_sequence) > max_seq_len:
        notes_input_sequence = notes_input_sequence[1:]
        durations_input_sequence = durations_input_sequence[1:]

    print(duration_result, note_result, sep=' \t')

    if note_result == 'START':
        break

print(f'Generated sequence of {len(prediction_output)} notes')
midi_stream = stream.Stream()

# create note and chord objects based on the values generated by the model
for pattern in prediction_output:
    note_pattern, duration_pattern = pattern
    # pattern is a chord
    if '.' in note_pattern:
        notes_in_chord = note_pattern.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(current_note)
            new_note.duration = duration.Duration(duration_pattern)
            new_note.storedInstrument = midi_instrument
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        midi_stream.append(new_chord)
    elif note_pattern == 'rest':
        # pattern is a rest
        new_note = note.Rest()
        new_note.duration = duration.Duration(duration_pattern)
        new_note.storedInstrument = midi_instrument
        midi_stream.append(new_note)
    elif note_pattern != 'START':
        # pattern is a note
        new_note = note.Note(note_pattern)
        new_note.duration = duration.Duration(duration_pattern)
        new_note.storedInstrument = midi_instrument
        midi_stream.append(new_note)

midi_stream = midi_stream.chordify()
midi_stream.write('midi', fp=output_filename)
