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