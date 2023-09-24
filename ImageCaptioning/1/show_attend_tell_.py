#!/usr/bin/env python
# coding: utf-8

# ## Show, Attend and Tell - архитектера  сети для создания подписей к фотографиям
# Статья: https://arxiv.org/pdf/1502.03044.pdf | https://www.tensorflow.org/tutorials/text/image_captioning?hl=ru

# #### Импорт библиотек

# In[1]:

import tensorflow_probability as tfp
import tensorflow as tf

from keras.layers import Input, Dense, LSTM, GRU, Dropout, Embedding, TextVectorization
from keras.layers import StringLookup, Add, Activation, concatenate, BatchNormalization
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


# ### Вспомогательные функции (красивой визуализации)

# In[2]:


def progress_bar(n_iter, n_total, prefix='Progress: ', suffix='', length=55, fill='█', lost='-'):
    percent = f"{100 * (n_iter / float(n_total)) :.1f}"
    filled_length = round(length * n_iter // n_total)
    bar = fill * filled_length + lost * (length - filled_length)
    print(f'\r{prefix}[{n_iter}/{n_total}] |{bar}| {percent}% {suffix}', end=' '*40)
    if n_iter == n_total:
        print()


# ### Пути до датасета

# In[3]:


PATH_DATASET = "D:\BigDataSets\ImageCaption_MS_COCO"
PATH_IMAGES = os.path.join(PATH_DATASET, "train2014")
PATH_IMAGES_EMB = os.path.join(PATH_DATASET, "train2014_InceptionEMB")
PATH_CAPTIONS = os.path.join(PATH_DATASET, "captions\captions_train2014.json")
SIZE_DATASET = 4_000


# ### Загрузка и подготовка датасета MS-COCO (13GB, 82K)

# In[4]:


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

# Select the first SIZE_DATASET all_image_paths from the shuffled set.
# Each image id has 5 captions associated with it, so that will lead to SIZE_DATASET*5 examples.
image_paths = all_image_paths[:SIZE_DATASET]
print("Train images:", len(image_paths))

captions = []
image_path_vector = []

for image_path in image_paths:
    caption_list = image_path_to_caption[image_path]
    captions.extend(caption_list)
    image_path_vector.extend([image_path] * len(caption_list))

del all_image_paths, image_path_to_caption


# In[5]:


rand_idx = random.randint(0, len(image_paths))

print(*captions[rand_idx*5:(rand_idx+1)*5], sep='\n')
Image.open(image_path_vector[rand_idx*5])


# ### Image embedding CNN model (InceptionResNetV2)
# - продвинутая модель от Google
# - заранее обученная модель для классификации (на ImageNet)
# - выделяет смысловые части из изображения

# In[6]:


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


# ## Кэширование вывода модели в память
# Один проход по всем изображениям CNN моделью и сохранение embedding-а в память (.npy) для текущей выбоки тренировочного набора

# In[7]:


# DELETE old embeddings
for file in os.listdir(PATH_IMAGES_EMB):
    filepath = os.path.join(PATH_IMAGES_EMB, file)
    os.unlink(filepath)

# IMAGE DATASET
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)  # Change batch_size to optimize

for image, path in image_dataset:  # for image, path in tqdm(image_dataset) - для кеширования мало ОЗУ
    batch_embedding = image_embedding_model(image)
    # Shape from (8, 8, 2048) to (64, 2048)
    batch_embedding = tf.reshape(batch_embedding, (batch_embedding.shape[0], -1, batch_embedding.shape[-1]))

    for p, b_e in zip(path, batch_embedding):
        p = p.numpy().decode("utf-8")
        p_e = os.path.join(PATH_IMAGES_EMB, p.split('\\')[-1])
        np.save(p_e, b_e.numpy())  # Save the embedding to file (img).npy


# ### Предварительная обработка и токенизация подписей (к фотографиям)

# In[8]:


# We will override the default standardization of TextVectorization to preserve "<>", we preserve for the <start> and <end>.
def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    inputs = tf.strings.regex_replace(inputs, r"[^\w\s<>]", "")
    return inputs

# Max word count for a caption.
max_seq_length = 30  # len(47, 12.45)
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


# In[9]:


word_to_index = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
index_to_word = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)
print(*tokenizer.get_weights()[0][:])


# ### Разделение данных для обучения, валидации и тестирования (70-15-15)

# In[10]:


img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(image_path_vector, caption_token_dataset):
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


# In[11]:


len(image_name_train), len(caption_train), len(image_name_val), len(caption_val), len(image_name_test), len(caption_test)


# ### Переменные (гипер-параметры) обучения

# In[12]:


# Feel free to change these parameters according to your system's configuration
EPOCHS = 30
BATCH_SIZE = 64
BUFFER_SIZE = 1000
NUM_STEPS = len(image_name_train) // BATCH_SIZE

embedding_dim = 256
units = 512
# Shape of the vector from InceptionV3 is (64, 2048). Two variables represent that vector shape.
features_shape = 2048
attention_features_shape = 64


# ### Создание основного датасета
# Оптимизация датасетов _<a href="https://www.tensorflow.org/guide/data_performance?hl=ru#prefetching">документация</a>_

# In[13]:


# Load the numpy files
def load_cache_img(img_name, cap):
    img_arr = np.load(os.path.join(PATH_IMAGES_EMB, img_name.decode('utf-8').split('\\')[-1] + '.npy'))
    return img_arr, cap

def make_dataset(image_block, caption_block, prefetch=True, cache=False):
    ds = tf.data.Dataset.from_tensor_slices((image_block, caption_block))
    
    # Use map to load the numpy files in parallel
    ds = ds.map(lambda item1, item2: tf.numpy_function(load_cache_img, [item1, item2], [tf.float32, tf.int64]),
                num_parallel_calls=tf.data.AUTOTUNE)     # Подгрузка изображений с диска
    
    # Shuffle and batch
    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)       # Перемешивать и выделять пакеты
    
    if prefetch:
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)   # Предварительная загрузка данных для скорости
    if cache:
        ds = ds.cache(filename='cache/file')             # Кеширование для скорости, но ОЗУ жрет и файл нужен
        
    return ds


dataset_train = make_dataset(image_name_train, caption_train, prefetch=True, cache=False)
dataset_valid = make_dataset(image_name_val, caption_val, prefetch=True, cache=False)
dataset_test = make_dataset(image_name_test, caption_test, prefetch=True, cache=False)

dataset_ = tf.data.Dataset.from_tensor_slices((image_name_train, caption_train))


# In[16]:


# Check dataset correction
for im_name, cap in dataset_.as_numpy_iterator():
    _im = Image.open(im_name.decode('utf-8')).show()
    print(*[index_to_word(w).numpy().decode('utf-8') for w in cap])
    break
    
# Check dataset shape
for im, cap in dataset_train.take(1):
    print(im.shape)
    print(cap.shape)
    print(cap[0])


# ### Модели (encoder, attention, decoder, model)

# In[29]:


# ENCODER
inp = Input(shape=(attention_features_shape, features_shape))
out_features = Dense(embedding_dim, activation='relu')(inp)
encoder = Model(inp, out_features, name='encoder')


# In[30]:


# ATTENTION (Bahdanau)
inp_features = Input(shape=(attention_features_shape, embedding_dim))

inp_hidden_state = Input(shape=(units,))
hidden_state_time_axis = tf.expand_dims(inp_hidden_state, axis=1)

features_hidden = Dense(units, activation='relu')(inp_features)                     # activation='relu'
hidden_state_time_axis = Dense(units, activation='relu')(hidden_state_time_axis)    # == +15% to speed loss train
attention_hidden_score = Activation('tanh')(features_hidden + hidden_state_time_axis)
attention_weights = tf.nn.softmax(Dense(1)(attention_hidden_score), axis=1)

context_vector = attention_weights * inp_features
context_vector = tf.reduce_sum(context_vector, axis=1)

attention = Model([inp_features, inp_hidden_state], [context_vector, tf.squeeze(attention_weights)], name='attention')


# In[31]:


# DECODER
inp_x_word = Input(shape=(1,))
inp_features = Input(shape=(attention_features_shape, embedding_dim))
inp_hidden_state = Input(shape=(units,))

x_word_emb = Embedding(vocabulary_size, embedding_dim)(inp_x_word)
context_vector, _attention_weights = attention([inp_features, inp_hidden_state])

x_concat = concatenate([tf.expand_dims(context_vector, 1), x_word_emb], axis=-1)
output, h_state = GRU(units, return_sequences=True, return_state=True,
                      recurrent_initializer="glorot_uniform")(x_concat)

output = tf.reshape(output, (-1, output.shape[-1]))
x = Dense(units, activation='relu')(output)  # activation='relu' == +15% to speed loss train
x = Dense(vocabulary_size, activation='softmax')(x)

decoder = Model([inp_x_word, inp_hidden_state, inp_features], [x, h_state, _attention_weights], name='decoder')


# In[ ]:


class ImageCaptioningModel(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_cross_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def loss_masked(self, real, pred):
        sub_loss = self.loss_cross_(real, pred)
        
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=sub_loss.dtype)
        sub_loss *= mask
        
        return tf.reduce_mean(sub_loss)

    # @tf.function
    def train_step(self, batch):
        image_tensor, target_caption = batch
        loss = 0.

        # initializing the hidden state for each batch because the captions are not related from image to image
        batch_size = target_caption.shape[0]
        hid_state = tf.zeros([batch_size, units], dtype=tf.int64)
        dec_input = tf.expand_dims([word_to_index('<start>')] * batch_size, 1)

        with tf.GradientTape() as tape:
            features = encoder(image_tensor)

            for i in range(1, max_seq_length):
                # passing the features through the decoder
                predictions, hid_state, _ = decoder([dec_input, hid_state, features])

                # loss += self.loss_masked(target_caption[:, i], predictions)
                loss += self.compiled_loss(target_caption[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target_caption[:, i], 1)

        total_loss = loss / max_seq_length

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss
    
    def call(self, image_embed, **kwargs):
        if len(image_embed.shape) == 2:
            image_embed = np.expand_dims(image_embed, axis=0)

        # initializing the hidden state for each batch because the captions are not related from image to image
        batch_size = target_caption.shape[0]
        hid_state = tf.zeros([batch_size, units])
        dec_inp = tf.expand_dims([word_to_index('<start>')] * batch_size, 1)
        result_id = [dec_inp]
        result_att = []

        features = encoder(image_embed)
        for i in range(1, max_seq_length):
            predictions, hid_state, attention_ = decoder([dec_input, hid_state, features])
            dec_input = tf.argmax(predictions, axis=-1)
            result_att.append(attention_)
        
        return result_id, result_att

    def build(self, **kwargs):
        super().build(self.encoder.input_shape)
    
    def save(self, fp, **kwargs):
        self.encoder.save(os.path.join(fp, "enc.h5"))
        self.decoder.save(os.path.join(fp, "dec.h5"))

    def load(self, fp):
        self.encoder = load_model(os.path.join(fp, "enc.h5"))
        self.decoder = load_model(os.path.join(fp, "dec.h5"))


# In[ ]:
def loss_masked(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    sub_loss = loss_cross(real, pred)

    mask = tf.cast(mask, dtype=sub_loss.dtype)
    sub_loss *= mask
    return tf.reduce_mean(sub_loss)


model = ImageCaptioningModel(encoder, decoder)
model.compile(tf.optimizers.Adam(), loss=loss_masked)

model.build()
model.summary()

model.fit(dataset_train, epochs=20, validation_data=dataset_valid)


# In[20]:

encoder.summary()
attention.summary()
decoder.summary()


# In[26]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
optimizer2 = tf.keras.optimizers.RMSprop(learning_rate=0.0007, rho=0.9, momentum=0.0)
loss_cross = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

rnn_output, h_state = GRU(units, return_sequences=True, return_state=True,
                          dropout=0.2, recurrent_dropout=0.2,
                          recurrent_initializer="glorot_uniform")(x_concat)

tokenizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
def loss_masked(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    sub_loss = loss_cross(real, pred)

    mask = tf.cast(mask, dtype=sub_loss.dtype)
    sub_loss *= mask
    return tf.reduce_mean(sub_loss)


# ### Сохранялки (от tensorflow) переделать

# In[ ]:


# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(encoder=encoder,
#                            decoder=decoder,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
# start_epoch = 0


# In[ ]:


# # Load save
# if ckpt_manager.latest_checkpoint:
#     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
#     ckpt.restore(ckpt_manager.latest_checkpoint)  # restoring the latest checkpoint in checkpoint_path


# ### Обучение

# In[27]:


@tf.function
def train_step(image_tensor, target_caption):
    loss = 0.

    # initializing the hidden state for each batch because the captions are not related from image to image
    batch_size = target_caption.shape[0]
    hid_state = tf.zeros([batch_size, units])
    dec_input = tf.expand_dims([word_to_index('<start>')] * batch_size, 1)

    with tf.GradientTape() as tape:
        features = encoder(image_tensor)

        for i in range(1, max_seq_length):
            # passing the features through the decoder
            predictions, hid_state, _ = decoder([dec_input, hid_state, features])

            loss += loss_masked(target_caption[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target_caption[:, i], 1)

    total_loss = loss / max_seq_length

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_loss


# In[28]:


loss_plot = []
for epoch in range(0, EPOCHS):  # start_epoch
    print(f'Epoch {epoch+1} started')
    start = time.time()
    total_loss = 0

    for (batch_idx, (image_tensor, target_caption)) in enumerate(dataset_train):
        t_loss = train_step(image_tensor, target_caption)
        
        progress_bar(batch_idx-1, NUM_STEPS, suffix=f'\tb_loss: {t_loss:.3f}')
        total_loss += t_loss

    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / NUM_STEPS)
    progress_bar(NUM_STEPS, NUM_STEPS, length=12, prefix=f'Epoch {epoch + 1}: ',
                 suffix=f'\tLoss: {total_loss / NUM_STEPS:.3f}\tTime: {time.time() - start:.2f} s')
    
    # Saving
    # ckpt_manager.save()


# In[ ]:




