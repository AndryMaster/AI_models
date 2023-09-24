from keras.datasets import boston_housing
from keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt
import numpy as np

def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
print(len(train_data), len(test_data))
print(train_data[10])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std
print(train_data[10])

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(train_data.shape[-1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)])

model.compile(optimizer=optimizers.RMSprop(),
              loss='mse',
              metrics=['mse', 'mae'])
model.build()

model.summary()

hist = model.fit(epochs=200, batch_size=40, x=train_data, y=train_labels,
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
mae = hist.history['mae']
val_mae = hist.history['val_mae']
plt.plot(epochs, mae, 'bo', label='Training mae')
plt.plot(epochs, val_mae, 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.legend()
plt.show()
