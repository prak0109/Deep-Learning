import tensorflow as tf

#node1 = tf.constant(5.0, tf.float32)
#node2 = tf.constant(4.0)

#print(node1,node2)

#sess = tf.Session()

#print(sess.run([node1,node2]))

#sess.close()

node1 = tf.constant(5.0, tf.float32)
node2 = tf.constant(6.0, tf.float32)

node3 = node1*node2

sess = tf.Session()

File_Writer = tf.summary.FileWriter('D:\Deep Learning Scripts\graph',sess.graph)
print(sess.run(node3))

sess.close()
print(type(node3))
