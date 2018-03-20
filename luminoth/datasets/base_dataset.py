import os
import tensorflow as tf
import sonnet as snt

from luminoth.datasets.exceptions import InvalidDataDirectory


class BaseDataset(snt.AbstractModule):
    def __init__(self, config, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        self._dataset_dir = config.dataset.dir
        self._num_epochs = config.train.num_epochs
        self._batch_size = config.train.batch_size
        self._split = config.dataset.split
        self._random_shuffle = config.train.random_shuffle
        self._seed = config.train.seed
        self._batch_size = config.train.batch_size

        self._fixed_resize = (
            'fixed_height' in config.dataset.image_preprocessing and
            'fixed_width' in config.dataset.image_preprocessing
        )
        if self._fixed_resize:
            self._image_fixed_height = (
                config.dataset.image_preprocessing.fixed_height
            )
            self._image_fixed_width = (
                config.dataset.image_preprocessing.fixed_width
            )

        self._total_queue_ops = 20

    def _build(self):
        # Find split file from which we are going to read.
        split_path = os.path.join(
            self._dataset_dir, '{}.tfrecords'.format(self._split)
        )
        if not tf.gfile.Exists(split_path):
            raise InvalidDataDirectory(
                '"{}" does not exist.'.format(split_path)
            )
        # String input producer allows for a variable number of files to read
        # from. We just know we have a single file.
        filename_queue = tf.train.string_input_producer(
            [split_path], num_epochs=self._num_epochs, seed=self._seed
        )

        # Define reader to parse records.
        reader = tf.TFRecordReader()
        _, raw_record = reader.read(filename_queue)

        values, dtypes, names = self.read_record(raw_record)

        # TODO add https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch too
        return tf.train.batch(
            values,
            capacity=100,
            batch_size=self._batch_size,
            dynamic_pad=True,
            num_threads=self._total_queue_ops)
