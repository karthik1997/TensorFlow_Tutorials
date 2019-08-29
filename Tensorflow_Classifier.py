from __future__ import absolute_import, division, print_function, unicode_literals

#installing TensorFlow
!pip install tensorflow==1.10.0

import tensorflow as tf

#loading mnist dataset

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train, x_test =  x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    
    
])


model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['acc'])


#Training and Evaluating the Model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)