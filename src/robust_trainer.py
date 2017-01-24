import tensorflow as tf
import numpy
from network.vgg_robust import VGG16Adversery, VGG16Robust


class RobustTrainer(object):
    """
    Train a network using robust objective
    """
    def __init__(self, learning_rate_net, learning_rate_adv):
        """
        Simple diagram:
         Images -> | FeatureNet -> FC | -------------------x-> loss
                                |_(no grad pass) -> |Adv| _|
        :return:
        """
        self.batch_size = 128

        self.actual_network = Learner()
        self.adverserial_network = Adversery()

        self.ph_images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ph_labels = tf.placeholder(tf.float32, [None, 10])
        self.ph_features = tf.placeholder(tf.float32, [None, 1024])
        self.ph_per_image_loss = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        real_net = VGG16Robust({'data': self.ph_images})
        self.features = real_net.layers['feat']
        real_pred = real_net.get_output()
        real_pred_sm = tf.nn.softmax(real_pred)

        real_net_vars = tf.trainable_variables()
        real_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in real_net_vars if 'bias' not in v.name ]) * 0.002

        self.per_image_loss = tf.nn.softmax_cross_entropy_with_logits(real_pred, self.ph_labels),
        self.real_loss = tf.reduce_mean(self.per_image_loss , 0)  + real_l2_loss
        self.real_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_net).minimize(self.real_loss)

        correct_prediction = tf.equal(tf.argmax(real_pred_sm, 1), tf.argmax(self.ph_labels, 1))
        self.real_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.summary.scalar("summary_real_training_accuracy", self.real_accuracy)
        tf.summary.scalar("summary_real_loss", self.real_loss)
        self.summaries = tf.summary.merge_all()
        self.test_summary = tf.summary.scalar("summary_real_test_accuracy", self.real_accuracy)

        adv_net = VGG16Adversery({'net_features': self.ph_features})
        adv_out = adv_net.get_output() # num images x 1
        self.adv_dist = tf.nn.softmax(adv_out,dim=0)

        net_vars = tf.trainable_variables()
        adv_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in net_vars
                                  if 'bias' not in v.name and 'adv' in v.name ]) * 0.002

        #sh_pred: num_examplesx1 adv_out num_example,1
        self.adv_loss = tf.reduce_mean(tf.multiply(self.ph_per_image_loss, self.adv_dist),0) - adv_l2_loss
        self.adv_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_adv).maximize(self.adv_loss)

    def assign_batch(self, large_batch):
        self.large_batch = large_batch

    @staticmethod
    def sample_with_replacement(large_batch, loss_estimates, batch_size, gamma=0.5):
        assert gamma < 1.0
        # bimodal probabilities
        num_examples = large_batch.shape[0]
        uniform_prob = 1.0 / num_examples
        uniform_prob_list = numpy.ones(num_examples) * uniform_prob
        bimodal_dist = (1-gamma) * loss_estimates + gamma * uniform_prob
        choices = numpy.random.choice(num_examples, batch_size, replace=True, p=bimodal_dist)
        small_batch = {'images':large_batch['images'][choices], 'labels':large_batch['labels'][choices]}
        return small_batch

    def learning_step(self, session, gamma):
        # this is the learning
        feat_values, per_im_loss = session.run([self.features, self.per_image_loss],
                                               feed_dict={self.ph_images: self.large_batch['images'],
                                                          self.ph_labels: self.large_batch['labels']})

        # update the discriminator
        loss_estimates, _ = session.run([self.adv_dist, self.adv_train_op],
                                        feed_dict={self.ph_features:feat_values,
                                                   self.ph_per_image_loss:per_im_loss})


        small_batch = RobustTrainer.sample_with_replacement(self.large_batch,
                                                            loss_estimates,
                                                            self.batch_size,
                                                            gamma)

        _ = session.run([self.real_train_op], feed_dict={self.ph_images: small_batch['images'],
                                                         self.ph_labels: small_batch['labels']})

    def summary_step(self, images, labels, session):
        summ, loss = session.run([self.summaries, self.real_loss],
                                 feed_dict={self.ph_images: images,
                                            self.ph_labels: labels})
        return loss, summ

    def test_step(self, images, labels, session):
        acc, summ = session.run([self.real_accuracy, self.test_summary],
                                feed_dict={self.ph_images: images,
                                           self.ph_labels: labels})
        return acc, summ