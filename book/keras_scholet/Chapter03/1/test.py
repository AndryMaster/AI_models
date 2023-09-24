from keras.datasets import reuters
from keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt
import numpy as np

def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data), len(test_data))
print(train_data[10])

train_data, test_data = vectorize_sequences(train_data), vectorize_sequences(test_data)
print(train_data[10])

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(word_index)

model = models.Sequential([
    layers.Dropout(0.1),
    layers.Dense(96, activation='elu', input_shape=(10_000,)),
    layers.Dropout(0.1),
    layers.Dense(64, activation='elu'),
    layers.Dropout(0.1),
    layers.Dense(46, activation='softmax')])

model.compile(optimizer=optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.build()

model.summary()

hist = model.fit(epochs=25, batch_size=512, x=train_data, y=train_labels,
                 validation_data=(test_data, test_labels))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training mae')
plt.plot(epochs, val_acc, 'b', label='Validation mae')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
