import keras.saving.object_registration
from keras.datasets import mnist
from keras import activations as act
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten

import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # / 255
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
X_train, X_test = X_train.reshape(*X_train.shape, 1), X_test.reshape(*X_test.shape, 1)  # 60_000(10_000), 28, 28, 1

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation=act.relu, input_shape=(28, 28, 1)))
model.add(MaxPooling2D())  # AveragePooling2D (хуже)
model.add(Conv2D(128, kernel_size=3, activation=act.relu))  # padding(same[0], valid[n])
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation=act.softmax))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))
# self_model.save('models/mnist1.keras')
# self_model = load_model('models/mnist1.keras')

# Test
def test(size=25):
    n = np.random.randint(0, y_test.shape[0], size=size)
    X_train_n = X_test[n]
    y_train_n = y_test[n]
    predict = model.predict(X_train_n, verbose=0)
    plt.figure(figsize=(10, 10))
    for i in range(size):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        i1, i2 = np.argmax(y_train_n[i]), np.argmax(predict[i])
        plt.title(f'{i1} -> {i2}', c='green' if i1 == i2 else 'red')
        plt.imshow(X_train_n[i], cmap=plt.cm.binary)
    plt.show()


for _ in range(5):
    test()
