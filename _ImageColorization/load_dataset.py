from skimage.color import rgb2lab
import numpy as np


def preprocess_image_old(image):
    image = image.resize((256, 256))
    image = np.array(image, dtype=np.float32)
    size = image.shape
    lab = rgb2lab(image/255)
    X, Y = lab[:, :, 0], lab[:, :, 1:]

    Y /= 128  # -1 to 1; x = 0 to 100
    X = X.reshape(1, size[0], size[1], 1)  # L      np.expand_dims(X, axis=[0, -1])
    Y = Y.reshape(1, size[0], size[1], 2)  # A, B   np.expand_dims(Y, axis=0)
    return X, Y, size


def clean_data(arr):
    array = np.array(arr, dtype=np.float32)
    array[:, :, :, 0] = array[:, :, :, 0] / 2.55
    array[:, :, :, 1:] = array[:, :, :, 1:] / 128 - 1
    return array


# Warning: memory over!
def load_data(directory='../dataset/color_img', train_range=(int, int), test_range=(int, int)):
    if max(max(train_range), max(test_range)) > 25000:
        assert IndexError('Dataset out of range')

    imgs_l = np.load(f'{directory}/l/gray_scale.npy')
    imgs_ab1 = np.load(f'{directory}/ab/ab/ab1.npy')
    imgs_ab2 = np.load(f'{directory}/ab/ab/ab2.npy')
    imgs_ab3 = np.load(f'{directory}/ab/ab/ab3.npy')
    imgs_ab = np.vstack([imgs_ab1, imgs_ab2, imgs_ab3])
    del imgs_ab1, imgs_ab2, imgs_ab3

    imgs = np.zeros((25000, 224, 224, 3), dtype=np.uint8)
    imgs[:, :, :, 0] = imgs_l
    imgs[:, :, :, 1:] = imgs_ab
    del imgs_l, imgs_ab

    train = clean_data(imgs[train_range[0]:train_range[1]])
    test = clean_data(imgs[test_range[0]:test_range[1]])
    print(imgs.shape, train.shape, test.shape)
    del imgs

    # print(train[0, 0])
    # print(np.max(train[:, :, :, 0]), np.min(train[:, :, :, 0]))
    # print(np.max(train[:, :, :, 1]), np.min(train[:, :, :, 1]))
    # print(np.max(train[:, :, :, 2]), np.min(train[:, :, :, 2]))
    return train, test
