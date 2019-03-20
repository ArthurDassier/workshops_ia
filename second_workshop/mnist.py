from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# get mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# get next 100 images
get_next_images, get_next_outs = mnist.train.next_batch(100)
# get test images
get_test_images = mnist.test.images
# get test labels
get_test_labels = mnist.test.labels

# true = résultats attendus / pred = prédictions du modèle
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(true, axis=1), predictions=tf.argmax(pred, 1))
accuracy = sess.run(acc_op, feed_dict={true: mnist.test.labels, pred: outs})
# accuracy = pourcentage de réussite du modèle
print(accuracy)
