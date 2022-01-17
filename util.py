import json
#import tensorflow as tf
import keras
from keras import layers, models
import os
import numpy as np
import base64
import cv2 as cv
__class_name_to_number = {}
__class_number_to_name = {}

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def create_model(className):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()  # let's have a look at our model so far

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(className)))
    model.summary()
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return img

def resize(img):
    width = 200
    height = 200 # keep original height
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized

def reurn_cropped_images(image):
    cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
    face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
    face_cascade = cv.CascadeClassifier(face_model)
    eye_cascade = cv.CascadeClassifier(eye_model)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    print(len(faces))
    cropped_images = []
    if len(faces) >= 1:
        for face in faces:
            (x, y, w, h) = face
            face_img = cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_face = face_img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(cropped_face)
            if len(eyes) >= 2:
                image_conv = resize(cropped_face)
                cropped_images.append(image_conv)
    return cropped_images

def load_model(className):

    model = create_model(className)
    checkpoint_path = "training_1/cp.ckpt"
    model.load_weights(checkpoint_path)
    return model

def classify_image(image,className):
    global __class_name_to_number
    global __class_number_to_name
    result = []
    with open("class_dictionary.json", "r") as f:

        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
    img = get_cv2_image_from_base64_string(image)
    chopped_images = reurn_cropped_images(img)
    if chopped_images == []:
        result.append({
            'class': class_number_to_name(5),
            'class_probability': str(0),
            'class_dictionary': __class_name_to_number
        })
        return result
    modeli = load_model(className)
    predict_images_mat = np.array(chopped_images)
    predictions = modeli.predict(predict_images_mat)
    max_values = []
    max_indexes = []
    if len(predictions)>1:
        for prediction in predictions:
            max_values.append(max(prediction))
            max_indexes.append(prediction.index(max(prediction)))
        maxVal = max(max_values)
        max_index = max_values.index(maxVal)
        the_index = max_indexes[max_index]
        if max_values[max_index] > 0.5:
            result.append({
                'class': class_number_to_name(the_index),
                #'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
                'class_dictionary': __class_name_to_number
            })
            return result
        else:
            result.append({
                'class': class_number_to_name(5),
                'class_probability': str(0),
                'class_dictionary': __class_name_to_number
            })
            return result
    else:
        if np.max(predictions) > 0.5:
            maxVal = np.max(predictions)
            ind = int(np.where(predictions == maxVal)[-1])
            result.append({
                'class': class_number_to_name(ind),
                'class_probability': str(maxVal),
                'class_dictionary': __class_name_to_number
            })
            return result
        else:
            result.append({
                'class': class_number_to_name(5),
                'class_probability': str(0),
                'class_dictionary': __class_name_to_number
            })
            return result

