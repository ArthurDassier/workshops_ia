import tensorflow as tf

x = [[0, 0],
     [1, 0],
     [0, 1],
     [1, 1]]

y = [[0], [1], [1], [0]]

entries = tf.placeholder(tf.float32, [None, 2])
hidden = tf.layers.dense(entries, 4, activation=tf.nn.relu)
outs = tf.layers.dense(hidden, 1)

labels = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outs, labels=labels), axis=0)
optimizer = tf.train.AdamOptimizer(0.01)
op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(loss, feed_dict={entries: x, labels: y})
        sess.run(op, feed_dict={entries: x, labels: y})
    outs = sess.run(tf.sigmoid(outs), feed_dict={entries: x})
    print(outs)
