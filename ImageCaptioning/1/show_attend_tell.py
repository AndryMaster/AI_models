import tensorflow as tf

from keras.layers import Input, Dense, LSTM, GRU, Dropout, Embedding, TextVectorization
from keras.layers import StringLookup, Add, Activation, concatenate
from keras.models import Model, load_model
import keras

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import collections
import random
import time
import json
import os
tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

PATH_DATASET = "D:\BigDataSets\ImageCaption_MS_COCO"
PATH_IMAGES = os.path.join(PATH_DATASET, "train2014")
PATH_IMAGES_EMB = os.path.join(PATH_DATASET, "train2014_InceptionV3")
PATH_CAPTIONS = os.path.join(PATH_DATASET, "captions\captions_train2014.json")


with open(PATH_CAPTIONS, 'r') as f:
    annotations = json.load(f)

image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    caption = f"<start> {val['caption']} <end>"
    image_path = os.path.join(PATH_IMAGES, f'COCO_train2014_{str(val["image_id"]).zfill(12)}.jpg')
    image_path_to_caption[image_path].append(caption)

all_image_paths = list(image_path_to_caption.keys())
random.shuffle(all_image_paths)  # shuffle dataset
print("All images:", len(all_image_paths))
# print(*all_image_paths[:10], sep='\n')

# Select the first 6_000 all_image_paths from the shuffled set.
# Each image id has 5 captions associated with it, so that will lead to 30_000 examples.
image_paths = all_image_paths[:6000]
print("Train images:", len(image_paths))

captions = []
image_path_vector = []

for image_path in image_paths:
    caption_list = image_path_to_caption[image_path]
    captions.extend(caption_list)
    image_path_vector.extend([image_path] * len(caption_list))

del all_image_paths, image_path_to_caption


print(*captions[:5], sep='\n')
Image.open(image_path_vector[0])

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
# image_model.summary()


new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_embedding_model = tf.keras.Model(new_input, hidden_layer)
print(hidden_layer.shape)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# DELETE old embeddings
for file in os.listdir(PATH_IMAGES_EMB):
    filepath = os.path.join(PATH_IMAGES_EMB, file)
    os.unlink(filepath)

# IMAGE DATASET
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(
    64)  # Change batch_size to optimize

for image, path in image_dataset:  # for image, path in tqdm(image_dataset) - для кеширования мало ОЗУ
    batch_embedding = image_embedding_model(image)
    # Shape from (8, 8, 2048) to (64, 2048)
    batch_embedding = tf.reshape(batch_embedding, (batch_embedding.shape[0], -1, batch_embedding.shape[-1]))

    for p, b_e in zip(path, batch_embedding):
        p = p.numpy().decode("utf-8")
        p_e = os.path.join(PATH_IMAGES_EMB, p.split('\\')[-1])
        np.save(p_e, b_e.numpy())  # Save the embedding to file (img).npy

# tf.keras.backend.clear_session()


# We will override the default standardization of TextVectorization to preserve "<>", we preserve for the <start> and <end>.
def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs, r"[!\"#$%&()*+.,-/:;=?@\^_`{|}~]", "")

# Max word count for a caption.
max_seq_length = 50
# Use the top 5000 words for a vocabulary.
vocabulary_size = 5000

# CAPTION DATASET
caption_dataset = tf.data.Dataset.from_tensor_slices(captions)

# TOKENIZER
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_seq_length,
    ngrams=None)
tokenizer.adapt(caption_dataset)

caption_token_dataset = caption_dataset.map(tokenizer)

word_to_index = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
index_to_word = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)
tokenizer.get_weights()

img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(image_path_vector, captions):
    img_to_cap_vector[img].append(cap)

# Create training, validation and test sets using an 70-15-15 split randomly.
image_keys = list(img_to_cap_vector.keys())
random.shuffle(image_keys)

slice_first = int(len(image_keys) * 0.7)
slice_second = int(len(image_keys) * 0.85)
img_name_train_keys, img_name_val_keys, img_name_test_keys = \
    image_keys[:slice_first], image_keys[slice_first:slice_second], image_keys[slice_second:]

# Train
image_name_train = []
caption_train = []
for img_name in img_name_train_keys:
    capt_len = len(img_to_cap_vector[img_name])
    image_name_train.extend([img_name] * capt_len)
    caption_train.extend(img_to_cap_vector[img_name])

# Validate
image_name_val = []
caption_val = []
for img_name in img_name_val_keys:
    capt_len = len(img_to_cap_vector[img_name])
    image_name_val.extend([img_name] * capt_len)
    caption_val.extend(img_to_cap_vector[img_name])

# Test
image_name_test = []
caption_test = []
for img_name in img_name_test_keys:
    capt_len = len(img_to_cap_vector[img_name])
    image_name_test.extend([img_name] * capt_len)
    caption_test.extend(img_to_cap_vector[img_name])

len(image_name_train), len(caption_train), len(image_name_val), len(caption_val), len(image_name_test), len(caption_test)


# Feel free to change these parameters according to your system's configuration
EPOCHS = 20
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
NUM_STEPS = len(image_name_train) // BATCH_SIZE
# Shape of the vector from InceptionV3 is (64, 2048) Two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


# Load the numpy files
def load_cache_img(img_name, cap):
    img_arr = np.load(PATH_IMAGES_EMB + img_name.decode('utf-8').split('\\')[-1] + '.npy')
    return img_arr, cap

dataset = tf.data.Dataset.from_tensor_slices([image_name_train, caption_train])

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_cache_img, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# ENCODER
inp = Input(shape=(attention_features_shape, features_shape))
out = Dense(embedding_dim, activation='relu')(inp)
encoder = Model(inp, out, name='encoder')

# ATTENTION
inp_features = Input(shape=(attention_features_shape, embedding_dim))
inp_hidden_state = Input(shape=(units,))

hidden_state_time_axis = tf.expand_dims(inp_hidden_state, axis=1)

features_hidden = Dense(units)(inp_features)
hidden_state_time_axis = Dense(units)(hidden_state_time_axis)
attention_hidden_layer = Activation('tanh')(features_hidden + hidden_state_time_axis)
attention_weights = Dense(1, activation='softmax')(attention_hidden_layer)

context_vector = attention_weights * inp_features
context_vector = tf.reduce_sum(context_vector, axis=1)

attention = Model([inp_features, inp_hidden_state], [context_vector, attention_weights], name='attention')

# DECODER
inp_x_word = Input(shape=(1,))
inp_features = Input(shape=(attention_features_shape, embedding_dim))
inp_hidden_state = Input(shape=(units,))

x_word_emb = Embedding(vocabulary_size, embedding_dim)(inp_x_word)
context_vector, _attention_weights = attention([inp_features, inp_hidden_state])

x_concat = concatenate([tf.expand_dims(context_vector, 1), x_word_emb], axis=-1)
output, h_state = GRU(units, return_sequences=True, return_state=True,
                      recurrent_initializer="glorot_uniform")(x_concat)

x = Dense(units)(output)
x = tf.reshape(x, (-1, x.shape[-1]))
x = Dense(vocabulary_size, activation='soft_max')(x)

decoder = Model([inp_x_word, inp_features, inp_hidden_state], [x, h_state, _attention_weights], name='decoder')

encoder.summary()
attention.summary()
decoder.summary()

optimizer = tf.keras.optimizers.Adam()
loss_cross = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_func(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    sub_loss = loss_cross(real, pred)

    mask = tf.cast(mask, dtype=sub_loss.dtype)
    sub_loss *= mask
    return tf.reduce_mean(sub_loss)


@tf.function
def train_step(image_tensor, target_caption):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    batch_size = target_caption.shape[0]
    # state_shape = list(decoder.outputs[1].shape)
    # state_shape[0] = target_caption.shape[0]
    hid_state = tf.zeros([batch_size, units])
    dec_input = tf.expand_dims([word_to_index('<start>')] * batch_size, 1)

    with tf.GradientTape() as tape:
        features = encoder(image_tensor)

        for i in range(1, max_seq_length):
            # passing the features through the decoder
            predictions, hid_state, _ = decoder(dec_input, features, hid_state)

            loss += loss_func(target_caption[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target_caption[:, i], 1)

    total_loss = loss / max_seq_length

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


loss_plot = []
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch_idx, (image_tensor, target_caption)) in enumerate(dataset):
        batch_loss, t_loss = train_step(image_tensor, target_caption)
        total_loss += t_loss

        if batch_idx % 100 == 0:
            average_batch_loss = batch_loss.numpy()
            print(f'Epoch {epoch + 1} Batch {batch_idx} Loss {batch_loss.numpy():.4f}')

    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / NUM_STEPS)

    # if epoch % 5 == 0:
    #     ckpt_manager.save()

    print(f'Epoch {epoch + 1} Loss {total_loss / NUM_STEPS:.6f}')
    print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')
