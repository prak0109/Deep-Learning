import tensorflow as tf

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

node3 = node1+node2

sess = tf.Session()

print(sess.run(node3,{node1:[1,3],node2:[2,4]}))

