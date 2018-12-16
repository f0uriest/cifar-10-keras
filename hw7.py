import functions as f
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# for running on cpu only, uncomment the following:
# num_cores = 4
# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
#         inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
#         device_count = {'CPU' : 1, 'GPU' : 0})
# tf.keras.backend.set_session(tf.Session(config=config))

# get training and validation data
data, labels = f.get_input_data([0,1,2,3,4])
val_data, val_labels = f.get_input_data([5])

# get labels
names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

# set up network
model = tf.keras.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4),
                        input_shape=(32, 32, 3)))
model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.MaxPool2D(pool_size=2, strides=2,
                           padding='valid', data_format='channels_last'))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.MaxPool2D(pool_size=2, strides=2,
                           padding='valid', data_format='channels_last'))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.MaxPool2D(pool_size=2, strides=2,
                           padding='valid', data_format='channels_last'))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same',
                        activation='elu', data_format='channels_last',
                        kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
model.add(layers.MaxPool2D(pool_size=2, strides=2,
                           padding='valid', data_format='channels_last'))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax',))

# set optimizer and training parameters
opt = tf.keras.optimizers.Adam(lr=.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# plot model architecture
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False,
                          to_file='model.png')
# model summary
print(model.summary())

# load old model from file
#model = tf.keras.models.load_model('model.h5')

# train
history = model.fit(data, labels, epochs=60, batch_size=100,
                    validation_data=(val_data, val_labels))

# save model and weights
model.save('model.h5')


# predict on validation data
labels_predict = model.predict_classes(val_data, batch_size=100)

# plot training data history
plot_train_history(history)

# generate and plot confusion matrix
labels_true = np.argmax(val_labels,axis=1)
plot_cm(labels_true,labels_predict)

# show a sampling of images and their predicted labels
labels = names[labels_predict[:16]]
f.save_imgs(val_data[:16],labels)
