import cv2 as cv
import os
import glob
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
keras = tf.keras
from tensorflow.keras import layers, models
import numpy as np

def resize(img):
    width = 200
    height = 200 # keep original height
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized

def get_images(location, format, lable,image_array,image_lable):
    files = glob.glob("" + location + "*." + format + "")
    for myFile in files:
        image = cv.imread(myFile)
        image_conv = resize(image)
        image_array.append(image_conv)  # append each image to array
        image_lable.append(lable)
    return image_array,image_lable





image_paths = ['players/train/CR','players/train/LM','players/train/PD','players/train/SA','players/train/SR']
test_paths = ['players/test/CR','players/test/LM','players/test/PD','players/test/SA','players/test/SR']
className = ["Christiano Ronaldo","Lionel Messi","Paulo Dybala","Sergio Aguero","Sergio Romero"]

im_arr = []
label_arr =[]
for count, x in enumerate(image_paths):
    im_arr,lable_arr= get_images(""+x+"/","jpg",count,im_arr,label_arr)


train_images_mat = np.array(im_arr)
train_labels_mat = np.array(lable_arr).transpose()

#add convolutional neural network layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()  # let's have a look at our model so far

#add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(className)))
model.summary()

#Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Save the model locarion in checkpoint_path
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

im_arr = []
label_arr =[]
for count, x in enumerate(test_paths):
    im_arr,lable_arr= get_images(""+x+"/","jpg",count,im_arr,label_arr)

test_images_mat = np.array(im_arr)
test_labels_mat = np.array(lable_arr).transpose()
#Train the model
history = model.fit(train_images_mat, train_labels_mat, epochs=4,
                    validation_data=(test_images_mat, test_labels_mat),
                    callbacks=[cp_callback])

#test accuracy of test data set
test_loss, test_acc = model.evaluate(test_images_mat,  test_labels_mat, verbose=2)

print(test_acc,test_loss)

#convert validation images into numpy matrix
'''predict_images_mat = np.array(image_array_val)
predict_labels_mat = np.array(label_array_val).transpose()
predictions = model.predict(predict_images_mat)

#Predict the validation data
predict =[]
for x in range(0,len(predictions),1):
    predict.append(np.argmax(predictions[x]))'''
print("sd")
