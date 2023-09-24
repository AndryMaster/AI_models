import tensorflow as tf
from keras.applications import MobileNetV2, vgg19
from keras.optimizers import Adam
from tensorflow.python import keras

# import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def prepare_img(image):
    return np.expand_dims(np.array(image, dtype=np.float32) / 127.5 - 1, axis=0)

def deprepare_img(image):
    return np.array(np.squeeze(image + 1) / 2)

def prepare_vgg19_img(image):
    return vgg19.preprocess_input(np.expand_dims(image, axis=0))

def deprepare_vgg19_img(image):
    image_copy = np.copy(np.squeeze(image))
    image_copy[:, :, :] += norm_means
    image_copy = image_copy[:, :, ::-1]
    return np.clip(image_copy.astype(np.uint8), 0, 255)

norm_means = np.array([103.939, 116.779, 123.680])

img = prepare_vgg19_img(Image.open('images/cats.jpg'))
img_style = prepare_vgg19_img(Image.open('images/style1.jpg'))

plt.subplot(1, 2, 1)
imshow(deprepare_vgg19_img(img), 'Content image')
plt.axis('off')
plt.subplot(1, 2, 2)
imshow(deprepare_vgg19_img(img_style), 'Style image')
plt.axis('off')
plt.show()
# sys.exit(1)

# content_layers = ['Conv_1_bn']
# style_layers = [
#     'block_1_depthwise_relu',
#     'block_4_depthwise_relu',
#     'block_7_depthwise_relu',
#     'block_10_depthwise_relu',
#     'block_13_depthwise_relu',
#     'block_16_depthwise_relu']
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# mobile_net2 = MobileNetV2(input_shape=(224, 224, 3), include_top=False, alpha=1)
# mobile_net2.trainable = False
# mobile_net2.summary()
vgg = vgg19.VGG19(include_top=False, weights="imagenet")
vgg.trainable = False
vgg.summary()

content_outputs = [vgg.get_layer(name).output_layer for name in content_layers]  # mobile_net2
style_outputs = [vgg.get_layer(name).output_layer for name in style_layers]  # mobile_net2
model_outputs = content_outputs + style_outputs

print(vgg.input, end='\n' * 2)  # mobile_net2
for m in model_outputs:
    print(m)

model = keras.models.Model(vgg.input, model_outputs)  # mobile_net2
model.summary()


def get_feature_representations():
    content_outputs_ = model(img)
    style_outputs_ = model(img_style)

    content_features_ = [content_layer[0] for content_layer in content_outputs_[:num_content_layers]]
    style_features_ = [style_layer[0] for style_layer in style_outputs_[num_content_layers:]]
    return content_features_, style_features_

@tf.function  # for fast +-
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

@tf.function  # for fast +-
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

@tf.function  # for fast +-
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    mat = tf.reshape(input_tensor, shape=[-1, channels])
    n = tf.shape(mat)[0]
    gram = tf.matmul(a=mat, b=mat, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def compute_loss(content_weight, style_weight, init_image, gram_style_features, content_features):
    model_outputs_ = model(init_image)

    style_output_features = model_outputs_[num_content_layers:]
    content_output_features = model_outputs_[:num_content_layers]

    content_score = style_score = 0
    weight_per_content_layer = 1 / len(content_layers)
    weight_per_style_layer = 1 / len(style_layers)

    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Get side loss
    loss = content_score * content_weight + style_score * style_weight
    return loss, content_score, style_score


num_iterations = 120
content_weight = 1000
style_weight = 0.015

content_features, style_features = get_feature_representations()
gram_style_features = [gram_matrix(style_layer) for style_layer in style_features]

init_image = tf.Variable(np.copy(img), dtype=tf.float32)
min_val, max_val = -norm_means, 255 - norm_means  # -1., 1.

optimizer = Adam(learning_rate=1, beta_1=0.99, epsilon=0.1)  # beta_1=0.99, epsilon=0.1
best_loss, best_img = np.inf, None
step = 10
img_history = np.zeros((num_iterations // step, *img.shape), dtype=np.uint8)

for i in range(1, num_iterations+1):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(content_weight=content_weight, style_weight=style_weight,
                                init_image=init_image, gram_style_features=gram_style_features,
                                content_features=content_features)

        total_loss, content_loss, style_loss = all_loss
        grads = tape.gradient(total_loss, init_image)
        optimizer.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_val, max_val)
        init_image.assign(clipped)

        if total_loss < best_loss:
            best_loss = total_loss
            best_img = init_image.numpy()

        if i % step == 0:
            img_history[i // step - 1] = deprepare_vgg19_img(init_image.numpy())

        print(f"Iteration: {i}\t({content_loss}; {style_loss})")


plt.figure(figsize=(8, 6))
for i in range(1, img_history.shape[0]+1):
    plt.subplot(3, 4, i)
    imshow(img_history[i-1], title=str(step * i))
    plt.axis('off')
plt.show()
print(f'Best_loss: {best_loss}')

best_img = Image.fromarray(deprepare_vgg19_img(best_img), "RGB")
best_img.save('real_images/result2.jpg')
