import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adagrad,Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#### Hyperparameters and variables ###

img_size = 256 ## to be defined later
epochs  = 50000
train_path = '/home/pranav/PycharmProjects/Fingerprint live detection/liveDet_data/Digital_Persona'
test_path = '/home/pranav/PycharmProjects/Fingerprint live detection/liveDet_data/Digital_Persona'
rotation_range  = 10

# Dataset Part ###

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=rotation_range,
        horizontal_flip=True,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(img_size, img_size),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

test_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=32    ,
        class_mode='categorical')

####  End dataset part ####



#####

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
#model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

#adgrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=1,
        validation_data=validation_generator,
        validation_steps=1,
        verbose=True,
        )

model.save_weights('model.h5')  # always save your weights after training or during training

