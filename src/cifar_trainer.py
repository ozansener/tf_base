import tensorflow as tf
from network.vgg import VGG16


class Cifar10Trainer(object):
    """
    Train a network using MNIST data or a dataset following 
    """
    def __init__(self):
        self.ph_images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ph_labels = tf.placeholder(tf.float32, [None, 10])

        net = VGG16({'data':self.ph_images})
        self.pred = net.get_output()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.ph_labels), 0)
        self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
        
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.ph_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.scalar_summary("training accuracy", self.accuracy)
        tf.scalar_summary("loss", self.loss)
        self.summaries = tf.summary.merge_all()

    def compute_accuracy(self, images, labels, session):
        acc = session.run(self.accuracy, feed_dict={self.ph_images: images, self.ph_labels:labels})
        return acc

    def train_step(self, images, labels, session):
        loss, summaries, _ = session.run([self.loss, self.summaries, self.train_op], feed_dict={self.ph_images: images, self.ph_labels: labels})
        return (loss, summaries)

