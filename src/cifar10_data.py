"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import glob

from six.moves import urllib
import numpy
import cPickle
SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'

def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    print(filepath)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        os.system('tar xvzf '+filename)
    return work_directory

def extract_data(work_directory, one_hot):
    print('Extracting', work_directory)
    data_files = glob.glob(work_directory+"/data_batch_*")
    num_images = len(data_files)*10000;
    images = numpy.zeros((num_images, 32,32,3))
    labels = numpy.zeros((num_images))
    for i,fil in enumerate(data_files):
        with open(fil, 'rb') as f:
            dat_dict = cPickle.load(f)
            ser_dat = dat_dict['data']
            images[i*10000:(i+1)*10000,:,:,0] = ser_dat[:,0:1024].reshape((-1,32,32)).astype('float32') - 125.0
            images[i*10000:(i+1)*10000,:,:,1] = ser_dat[:,1024:2048].reshape((-1,32,32)).astype('float32') - 123.0
            images[i*10000:(i+1)*10000,:,:,2] = ser_dat[:,2048:3072].reshape((-1,32,32)).astype('float32') - 114.0
            labels[i*10000:(i+1)*10000] = dat_dict['labels']


    if one_hot:
        return images, dense_to_one_hot(labels)
    else:
        return images, labels


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.astype(int).ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, one_hot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()

    DATA_SET_NAME = "cifar-10-python.tar.gz"
    local_file = maybe_download(DATA_SET_NAME,train_dir)
    im,l = extract_data(train_dir, one_hot)
    data_sets.train = DataSet(im,l)

    return data_sets
