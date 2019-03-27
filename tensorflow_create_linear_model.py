import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.Variable([-3.0],tf.float32)
b = tf.Variable([3.0],tf.float32)

x = tf.placeholder(tf.float32)

Linear_model = a*x + b

y = tf.placeholder(tf.float32)

#training data

squared_deltas = tf.square(Linear_model - y)

loss = tf.reduce_sum(squared_deltas)

#Optimizing Data

optimizer =tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    
#print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

print(sess.run([a,b]))

