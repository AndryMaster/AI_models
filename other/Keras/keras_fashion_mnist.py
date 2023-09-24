from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import activations as activate
from keras.optimizers import Adam  # SGD, RMSprop, Adagrad, Adamax, Nadam

import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, test_images = train_images.reshape(*train_images.shape, 1) / 255., \
                            test_images.reshape(*test_images.shape, 1) / 255.  # 60_000(10_000), 28, 28, 1
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

optimizer = Adam(learning_rate=0.001)
model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation=activate.relu),
    Dense(10, activation=activate.softmax),
])
model2 = Sequential([
    Conv2D(32, kernel_size=3, activation=activate.relu, input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Conv2D(64, kernel_size=3, padding='same', activation=activate.relu),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.15),
    Dense(64, activation=activate.sigmoid),
    Dense(10, activation=activate.softmax),
])

model = model2
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.build()
model.summary()

model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10)
# self_model.save('models/mnist_fashion2.h5', save_format='h5')
# self_model = load_model('models/mnist_fashion2.h5')

# Test
def test(images, labels):
    n = np.random.randint(0, test_labels.shape[0], size=25)
    test_img = images[n]
    test_lab = labels[n]
    predict = model.predict(test_img, verbose=0)
    plt.figure(figsize=(14, 14))
    for i in range(len(n)):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        pred = np.argmax(predict[i])
        label = test_lab[i]
        plt.title(f'{class_names[label]} {round(predict[i][pred] * 100)}% ({class_names[pred]})',
                  c='green' if label == pred else 'red')
        plt.imshow(test_img[i], cmap=plt.cm.binary)
    plt.show()


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Accuracy: {test_acc}')

for _ in range(4):
    test(test_images, test_labels)
