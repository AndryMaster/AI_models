from imageai.Detection import ObjectDetection
import matplotlib.pyplot as plt
import os

execution_path = os.getcwd()

detection = ObjectDetection()
detection.setModelTypeAsYOLOv3()
detection.setModelPath(os.path.join(execution_path, 'weights', 'yoloV3.h5'))
detection.loadModel()

custom = detection.CustomObjects(person=True, dog=True)  # .detectCustomObjectsFromImage()
path_img = os.path.join(execution_path, 'images', 'cars2.jpg')
path_save = os.path.join(execution_path, 'new_image.jpg')

detections, extract_objects = detection.detectObjectsFromImage(
    input_image=path_img,
    output_image_path=path_save,
    extract_detected_objects=False,
    minimum_percentage_probability=15)


# plt.figure(figsize=(4, 4))
# plt.title('Result')
# plt.imshow(plt.imread(path_save))
# plt.grid(False)
# plt.show()
