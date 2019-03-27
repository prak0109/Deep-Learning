import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  #28*28 images of hand-written digit 0-9

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test, axis =1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

plt.imshow(x_train[0],cmap = plt.cm.binary)
plt.show()
