import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory

from CycleGAN import CycleGAN


def show_ds_samples(ds_zip, count=1):
    for i in range(count):
        a, b = next(ds_zip)
        plt.imshow(a[0] * 0.5 + 0.5)
        plt.show()
        plt.imshow(b[0] * 0.5 + 0.5)
        plt.show()

tanh, sigmoid = lambda img: img / 127.5 - 1, lambda img: img / 255.

# Params
SAMPLES_PER_EPOCH = 450
BATCH_SIZE = 1
EPOCHS = 10
SIZE = 256

# interpolation='bilinear'
dataset_apple = image_dataset_from_directory('dataset/apple2orange/testA', labels=None, batch_size=BATCH_SIZE,
                                             crop_to_aspect_ratio=True, shuffle=True, image_size=(SIZE, SIZE))
dataset_orange = image_dataset_from_directory('dataset/apple2orange/testB', labels=None, batch_size=BATCH_SIZE,
                                              crop_to_aspect_ratio=True, shuffle=True, image_size=(SIZE, SIZE))
dataset_apple = dataset_apple.map(tanh)
dataset_orange = dataset_orange.map(tanh)
dataset_combined = (dataset_apple, dataset_orange)
# show_ds_samples(zip(dataset_apple, dataset_orange))

model = CycleGAN(
    input_dim=(SIZE, SIZE, 3),
    learning_rate=0.0002,
    buffer_max_length=50,
    weight_validation=1.6,
    weight_reconstr=10,
    weight_id=2.6,
    generator_type='unet',
    gen_n_filters=36,   # 36 32
    disc_n_filters=44,  # 44 48
    max_gen_count=200,             # Count images in dataset (if value more: StopIterationError)
    data_loader=dataset_combined,  # zip and next -> ends
    summary=True,
)

model.train(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    count_batches=SAMPLES_PER_EPOCH // BATCH_SIZE,)

# model.load_weights('models/ap2or_unet-36-44-model.h5')
# model.sample_images()
