from robust_trainer import RobustTrainer
import cifar10_data
import tensorflow as tf
from datetime import datetime
import click
import json
import numpy

def read_params(file_name):
    class Param(object):
        pass

    params = Param()
    with open(file_name) as json_data:
        d = json.load(json_data)
        params.batch_size = d['batch_size']
        params.learning_rate = d['learning_rate']
        params.dropout = d['dropout']
    str_v = "robust_learning_rate_{}__batch_size_{}__dropout_{}".format(params.batch_size, params.learning_rate, params.dropout)

    return params, str_v

@click.command()
@click.option('--hold_out', default=0, help='Training data size.')
@click.option('--dev_name', default='/gpu:0', help='Device name to use.')
@click.option('--sample/--no-sample', default=True)
def train(hold_out, dev_name, sample):
    hold_out_s = int(hold_out)
    print hold_out_s
    train_data = cifar10_data.read_data_sets('./data/', one_hot=True, hold_out_size=hold_out_s)

    params, str_v = read_params('settings.json')

    num_b = 5 * numpy.ceil( train_data.train.images.shape[0] / ((1.0)*params.batch_size) )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sesh:
        robust_cifar_train = RobustTrainer(params.learning_rate, params.learning_rate, dev_name)
        # no need to initialize anything since ve are simply re-loading the data
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sesh, 'models/cifar10_robust_{}_model_hold_out_{}__{}-{}'.format(sample,hold_out_s,str_v, batch_id))
        # here options, running the adv with more data (validation), learn adv with more data from scratch
        sum_writer = tf.summary.FileWriter("./dumps_robust__sample_{}__hold_out_{}__{}/".format(sample,hold_out_s,str_v), sesh.graph)

        robust_cifar_train.assign_batch(
        ap, gt = robust_cifar_train.active_sample(sesh,
                                                  {'images': train_data.holdout.images,
                                                   'labels': train_data.holdout.labels}, 5000)

        print ap
        print gt

if __name__ == '__main__':
    train()
