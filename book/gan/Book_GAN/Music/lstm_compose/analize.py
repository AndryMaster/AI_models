import numpy as np
from music21 import converter, note, chord

file1 = '../dataset/midi_songs_1/FFIX_Piano.mid'
file2 = '../dataset/midi_songs_2/cs1-2all.mid'
original_score = converter.parse(file2).chordify()

# print(original_score.show('text'))

notes = []
durations = []
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

# print('duration', 'pitch')
# for n, d in zip(notes, durations):
#     print(d, n, sep=' \t')

lr = 0.002
for e in range(1, 75):
    lr = np.linspace(0.0016, 0.0004, 75)[e]
    print(lr)
