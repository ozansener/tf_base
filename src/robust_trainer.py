import tensorflow as tf
import numpy
from network.vgg_robust import VGG16Adversery, VGG16Robust
import pdb

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
        dim = 512
        self.ph_images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ph_labels = tf.placeholder(tf.float32, [None, 10])
        self.ph_features = tf.placeholder(tf.float32, [None, dim])
        self.ph_per_image_loss = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase = tf.placeholder(tf.bool, name='phase')

        real_net = VGG16Robust({'data': self.ph_images},phase=self.phase)
        self.features = tf.reshape(real_net.layers['feat'],[-1, dim])
        real_pred = real_net.get_output()
        real_pred_sm = tf.nn.softmax(real_pred)

        real_net_vars = tf.trainable_variables()
        real_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in real_net_vars if 'bias' not in v.name ]) * 0.002

        self.per_image_loss = tf.nn.softmax_cross_entropy_with_logits(real_pred, self.ph_labels)
        self.real_loss = tf.reduce_mean(self.per_image_loss , 0)  + real_l2_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.real_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_net).minimize(self.real_loss)

        correct_prediction = tf.equal(tf.argmax(real_pred_sm, 1), tf.argmax(self.ph_labels, 1))
        self.accuracy_per_im = tf.cast(correct_prediction, "float")
        self.real_accuracy = tf.reduce_mean(self.accuracy_per_im)

        tf.summary.scalar("summary_real_training_accuracy", self.real_accuracy)
        tf.summary.scalar("summary_real_loss", self.real_loss)
        tf.summary.scalar("max per image loss in batch", tf.reduce_max(self.per_image_loss))
        self.summaries = tf.summary.merge_all()
        self.test_summary = tf.summary.scalar("summary_real_test_accuracy", self.real_accuracy)

        adv_net = VGG16Adversery({'net_features': self.ph_features},phase=self.phase)
        adv_out = adv_net.get_output() # num images x 2
        self.a = adv_out
        # this is for observation only, softmax becomes numerically unstable if this is greater than a few thousands
        # note(ozan) never ever use this without a batch-norm, batch-norm stablize the hell out of it
        
        # Todo(ozan) figure out what is going on here
        #max_adv_out = tf.reduce_max(adv_out)
        #self.abc = adv_out
        #self.adv_dist = tf.transpose(tf.nn.softmax(tf.transpose(adv_out)))

        net_vars = tf.trainable_variables()
        # this is a regularizer, a small weight decay
        adv_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in net_vars
                                  if 'bias' not in v.name and 'adv' in v.name ]) * 0.001

        self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(adv_out, self.ph_per_image_loss)) + adv_l2_loss
        #self.adv_loss = tf.reduce_sum(tf.multiply(self.ph_per_image_loss, self.adv_dist)) - adv_l2_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.adv_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_adv).minimize(self.adv_loss)
        self.adv_summ = tf.summary.merge([tf.summary.histogram("histogram of activations", adv_out), 
                                          #tf.summary.histogram("histogram_of_expected_loss", self.adv_dist), 
                                          tf.summary.histogram("histogram of input", self.ph_features),
                                          #tf.summary.scalar("max of adverserial output before softmax", max_adv_out),
                                          tf.summary.scalar("adverserial loss", self.adv_loss)])
    
    def assign_batch(self, large_batch):
        self.large_batch = large_batch

    @staticmethod
    def sample_with_replacement(large_batch, loss_estimates, batch_size, gamma=0.5):
        assert gamma <= 1.0
        # bimodal probabilities
        num_examples = large_batch['images'].shape[0]
        uniform_prob = 1.0 / num_examples
        uniform_prob_list = numpy.ones((num_examples,1)) * uniform_prob
        bimodal_dist = (1-gamma) * loss_estimates.reshape((num_examples,1)) + gamma * uniform_prob_list
        bimodal_dist_flat = bimodal_dist[:,0]
        choices = numpy.random.choice(num_examples, batch_size, replace=True, p=bimodal_dist_flat)
        small_batch = {'images':large_batch['images'][choices], 'labels':large_batch['labels'][choices]}
        return small_batch

    def learning_step(self, session, gamma,fn):
        feat_values, per_im_loss_dd, ll = session.run([self.features, self.per_image_loss, self.accuracy_per_im],
                                               feed_dict={self.ph_images: self.large_batch['images'],
                                                          self.ph_labels: self.large_batch['labels'],self.phase:0})

        #per_im_loss = per_im_loss_d.reshape((per_im_loss_d.shape[0],1))
        per_im_loss_d = 1.0 - ll.astype(numpy.float32)
        per_im_loss = per_im_loss_d.reshape((per_im_loss_d.shape[0],1))

        ce_lab = numpy.concatenate((per_im_loss, 1.0-per_im_loss), axis=1)

        adv_sum = session.run(self.adv_summ, feed_dict={self.ph_features:feat_values,
                                                                                self.ph_per_image_loss:ce_lab,
                                                                                self.phase:0})


        loss_estimates = session.run(tf.nn.softmax(self.a), feed_dict={self.ph_features:feat_values,
                                                                                self.phase:0})


        if fn>500: 
            _ = session.run([self.adv_train_op], feed_dict={self.ph_features:feat_values,
                                                                                self.ph_per_image_loss:ce_lab,
                                                                                self.phase:1})

        small_batch = RobustTrainer.sample_with_replacement(self.large_batch,
                                                            loss_estimates[:,0],
                                                            self.batch_size,
                                                            gamma)
        #print ll

        _ = session.run([self.real_train_op], feed_dict={self.ph_images: small_batch['images'],
                                                         self.ph_labels: small_batch['labels']})
        return adv_sum

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
