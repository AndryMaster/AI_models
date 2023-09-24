from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, Bidirectional
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

N = 12_000
train = 10_000
data = np.array([np.sin(x/20) for x in range(N)]) + 0.1 * np.random.randn(N)
# plt.plot(dataset[:170])
# plt.show()

off = 3
length = 2 * off + 1
X = np.array([np.diag(np.hstack((data[i:i+off], data[i+off+1:i+length]))) for i in range(N-length)])
Y = data[off:N-off-1]
print(X.shape, Y.shape)


model = Sequential([
    Input(shape=(length-1, length-1)),
    Bidirectional(GRU(2)),
    Dense(1, activation='linear')])

model.compile(optimizer=Adam(0.003), loss=mean_squared_error, metrics=['accuracy'])
model.summary()

history = model.fit(X[:train], Y[:train], validation_data=(X[train:], Y[train:]), epochs=10, batch_size=32)

# # Plot
# plt.figure(figsize=(10, 4))
# plt.plot(hist.hist['loss'], label='Training Loss', x='green', lw='2')
# plt.plot(hist.hist['val_loss'], label='Validation Loss', x='orangered', lw='2')
# plt.title('Loss', loc='left', fontsize=16)
# plt.xlabel("Epochs")
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Testing
train_data = data
M = 200
XX = np.zeros(M)
XX[:off] = data[:off]
for i in range(M-off-1):
    x = np.diag(np.hstack((XX[i:i+off], train_data[i+off+1:i+length])))
    x = np.expand_dims(x, axis=0)
    y = model.predict(x, batch_size=1, verbose=0)
    XX[i+off+1] = y

plt.plot(XX[:M])
plt.plot(train_data[:M])
plt.show()
