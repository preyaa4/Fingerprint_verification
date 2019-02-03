from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adagrad,Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from  tflearn.data_utils import image_preloader
import tensorflow as tf



def build_model():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))  # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


model_name = '/home/pranav/PycharmProjects/Fingerprint_verification/model.h5'
train_path = '/home/pranav/PycharmProjects/Fingerprint live detection/liveDet_data/Digital_Persona_Original'

model = build_model()

model.load_weights(model_name)

test_datagen = ImageDataGenerator(rescale=1./255)


test_generator = test_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')

print(model.metrics_names)
_,accuracy=model.evaluate_generator(test_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
print(accuracy)


######### getting true positives tp, true negatives tn, false positives fp and false negatives fn #########

_,y_true = test_generator.next()
#Confution Matrix and Classification Report
y_true = np.argmax(y_true, axis=1)
Y_pred = model.predict_generator(test_generator,steps=1)
y_pred = np.argmax(Y_pred, axis=1)


tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
print ("tn: ", tn,"\nfp: ", fp, "\nfn: ", fn, "\ntp: ", tp )

'''false_negative =0                       #actually live predicts fake
false_positive = 0                      #actually fake predicts live
true_positive=0                         #actually live predicts live
true_negative=0                         #actually fake predicts fake


for i in range(y_pred.shape[0]):
    if(y_true[i] == 0 and y_pred[i] == 0):
        true_negative=true_negative+1
    elif (y_true[i] == 0 and y_pred[i] == 1):
        false_positive=false_positive+1
    elif (y_true[i] == 1 and y_pred[i] == 0):
        false_negative = false_negative + 1
    elif (y_true[i] == 1 and y_pred[i] == 1):
        true_positive = true_positive + 1

print("fn, fp, tp, tn")
print (false_negative,false_positive,true_positive,true_negative)


print(y_true,y_pred)'''