import tensorflow as tf
from tensorflow import keras


def create_nn_model():
	if keras.image_data_format() == 'channels_first':
	    input_shape = (3, img_width, img_height)
	else:
	    input_shape = (img_width, img_height, 3)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
		      optimizer=tf.train.AdamOptimizer(),
		      metrics=['accuracy'])

	return model
