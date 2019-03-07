from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

entries = tf.placeholder(tf.float32, [None, 784])
hidden = tf.layers.dense(entries, 512, activation=tf.nn.relu)
outs = tf.layers.dense(hidden, 10)

labels = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outs, labels=labels), axis=0)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

true = tf.placeholder(tf.float32, shape=[None, 10], name='true')
pred = tf.placeholder(tf.float32, shape=[None, 10], name='pred')
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(true, axis=1), predictions=tf.argmax(pred, 1))

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(global_init)
    sess.run(local_init)
    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        loss_value = sess.run(loss, feed_dict={entries: batch_xs, labels: batch_ys})
        sess.run(optimizer, feed_dict={entries: batch_xs, labels: batch_ys})
    outs = sess.run(tf.sigmoid(outs), feed_dict={entries: mnist.test.images})
    accuracy = sess.run(acc_op, feed_dict={true: mnist.test.labels, pred: outs})
    print(accuracy)
