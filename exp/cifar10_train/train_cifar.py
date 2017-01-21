from cifar_trainer import Cifar10Trainer
import cifar10_data
import tensorflow as tf
from datetime import datetime

train_data = cifar10_data.read_data_sets('./data/', one_hot=True)
test_im = train_data.test.images
test_l = train_data.test.labels


def get_batch():
    im, l = train_data.train.next_batch(128)
    return im, l

with tf.Session() as sesh:
    cifar_train = Cifar10Trainer()
    print 'Initial Variable List:'
    print [tt.name for tt in tf.trainable_variables()]
    init = tf.global_variables_initializer()
    sesh.run(init)
    saver = tf.train.Saver(max_to_keep=200)
    sum_writer = tf.summary.FileWriter("./dumps/", sesh.graph)

    for batch_id in range(200000):
        im, l = get_batch()
        loss, summ, top = cifar_train.train_step(im, l, sesh)
        sum_writer.add_summary(summ, batch_id)
        # save every 100th batch model
        if batch_id % 500 == 0:
            saver.save(sesh, 'models/cifar10_model', global_step=batch_id)
            acc, test_sum = cifar_train.test_step(test_im, test_l, sesh)
            print "{}: step{}, test acc {}".format(datetime.now(), batch_id, acc)
            sum_writer.add_summary(test_sum, batch_id)

        if batch_id % 100 == 0:
            print "{}: step {}, loss {}".format(datetime.now(), batch_id, loss)

    #acc = cifar_train.compute_accuracy(test_im, test_l, sesh)
    #print 'Final Accuracy: ', acc
