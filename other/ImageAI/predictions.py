from imageai.Prediction import ImagePrediction
import matplotlib.pyplot as plt
import os

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, 'weights', 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'))
prediction.loadModel()
# setModelTypeAsResNet, setModelTypeAsDenseNet, setModelTypeAsSqueezeNet, setModelTypeAsInceptionV3

# results = prediction.predictMultipleImages(all_imgs, result_count_per_image=7)
all_imgs = os.listdir(os.path.join(execution_path, 'images'))
all_path_imgs = [os.path.join(execution_path, 'images', path_img) for path_img in all_imgs]
print(all_imgs)
for i, path_img in enumerate(all_path_imgs):
    predictions, probabilities = prediction.predictImage(path_img, result_count=7)
    print('-' * 20, all_imgs[i], '-' * 20)
    for each_prediction, each_probability in zip(predictions, probabilities):
        print(f'{each_prediction}\t({each_probability})')

plt.figure(figsize=(9, 9))
for i, path_img in enumerate(all_path_imgs):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(all_imgs[i])
    plt.imshow(plt.imread(path_img))
    plt.grid(False)
plt.show()
