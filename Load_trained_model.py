import cv2 as cv
import os
import glob
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def resize(img):
    width = 200
    height = 200 # keep original height
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized

def get_images(location, format):
    image_array = []
    files = glob.glob("" + location + "*." + format + "")
    cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
    face_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    eye_model = os.path.join(cv2_base_dir, 'data/haarcascade_eye.xml')
    face_cascade = cv.CascadeClassifier(face_model)
    eye_cascade = cv.CascadeClassifier(eye_model)
    for myFile in files:
        image = cv.imread(myFile)
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        print(len(faces))
        if len(faces) >= 1:
            for face in faces:
                (x, y, w, h) = face
                face_img = cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cropped_face = face_img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(cropped_face)
                if len(eyes) >= 2:
                    image_conv = resize(cropped_face)
                    image_array.append(image_conv)# append each image to array
                else:
                    print("Bad Image")
        else:
            print("Bad Image")
    return image_array


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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
className = ["CHristiano Ronaldo","Lionel Messi","Paulo Dybala","Sergio Aguero","Sergio Romero"]
model = create_model(className)
checkpoint_path = "training_1/cp.ckpt"

#load weights into the untrained model with the weights stored in checkpoint path
model.load_weights(checkpoint_path)

im_arr = get_images("pred/","jpg")
predict_images_mat = np.array(im_arr)
#test accuraacy of the test dataset
predictions = model.predict(predict_images_mat)
print(predictions)