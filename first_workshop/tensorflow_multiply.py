import tensorflow as tf

e = [[2, 2]]
f = [[3]]

a = tf.placeholder(tf.float32, [None, 2])
b = tf.placeholder(tf.float32, [None, 1])

mul = tf.math.multiply(a, b)

init = tf.init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(mul, feed_dict={a: e, b: f}))
