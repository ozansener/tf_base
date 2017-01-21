from cifar_trainer import Cifar10Trainer
import cifar10_data
import tensorflow as tf
from datetime import datetime
import click
import json

def read_params(file_name):
    class Param(object):
        pass

    params = Param()
    with open(file_name) as json_data:
        d = json.load(json_data)
        params.batch_size = d['batch_size']
        params.learning_rate = d['learning_rate']
    str_v = "learning_rate_{}__batch_size_{}".format(params.batch_size, params.learning_rate)

    return params, str_v

@click.command()
@click.option('--hold_out', default=0, help='Training data size.')
def train(hold_out):
    hold_out_s = int(hold_out)
    train_data = cifar10_data.read_data_sets('./data/', one_hot=True, hold_out_size=hold_out_s)

    test_im = train_data.test.images
    test_l = train_data.test.labels

    params, str_v = read_params('settings.json')

    with tf.Session() as sesh:
        cifar_train = Cifar10Trainer(learning_rate=params.learning_rate)
        print 'Initial Variable List:'
        print [tt.name for tt in tf.trainable_variables()]
        init = tf.global_variables_initializer()
        sesh.run(init)
        saver = tf.train.Saver(max_to_keep=200)
        sum_writer = tf.summary.FileWriter("./dumps_hold_out_{}__{}/".format(hold_out_s,str_v), sesh.graph)

        for batch_id in range(200000):
            im, l = train_data.train.next_batch(params.batch_size)
            loss, summ, top = cifar_train.train_step(im, l, sesh)
            sum_writer.add_summary(summ, batch_id)
            # save every 100th batch model
            if batch_id % 500 == 0:
                saver.save(sesh, 'models/cifar10_model_hold_out_{}__{}'.format(hold_out_s,str_v), global_step=batch_id)
                acc, test_sum = cifar_train.test_step(test_im, test_l, sesh)
                print "{}: step{}, test acc {}".format(datetime.now(), batch_id, acc)
                sum_writer.add_summary(test_sum, batch_id)

            if batch_id % 100 == 0:
                print "{}: step {}, loss {}".format(datetime.now(), batch_id, loss)

if __name__ == '__main__':
    train()
