# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import numpy as np
import time
import os
import argparse

import tensorflow as tf

import cifar10

def parse_args():

    parser = argparse.ArgumentParser(description='CIFAR-10 training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--conv1-channels',
                        dest    = 'conv1_channels',
                        help    = 'number of channels in first convolutional layer',
                        default = 64,
                        type    = int)

    parser.add_argument('--initial-lr',
                        dest    = 'initial_lr',
                        help    = 'initial learning rate for fine tunning',
                        default = 0.01,
                        type    = float)

    parser.add_argument('--fine-tune-steps',
                        dest    = 'fine_tune_steps',
                        help    = 'number of steps for fine tunning',
                        default = 12500,
                        type    = int)

    parser.add_argument('--load-file',
                        dest    = 'load_file',
                        help    = 'file with pretrained model',
                        default = None,
                        type    = str)

    parser.add_argument('--save-file',
                        dest    = 'save_file',
                        help    = 'file with trained model',
                        default = None,
                        type    = str)

    args = parser.parse_args()

    return args

# Parse script arguments
args = parse_args()

INITIAL_LR = args.initial_lr
FINE_TUNE_STEPS = args.fine_tune_steps
MODEL_SAVE_FILE = args.save_file
MODEL_LOAD_FILE = args.load_file
NUM_CONV1_CHANNELS = args.conv1_channels
if NUM_CONV1_CHANNELS == 64:
    MODEL_DIR = r'./data/models/baseline'
else:
    MODEL_DIR = os.path.join(r'./data/models/', str(NUM_CONV1_CHANNELS))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', MODEL_DIR,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, NUM_CONV1_CHANNELS)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def train_custom():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, NUM_CONV1_CHANNELS)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, initial_learning_rate=INITIAL_LR)

    with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # begin enqueue thread
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_dict = np.load(MODEL_LOAD_FILE, encoding='latin1').item()

        for scope in ['conv1', 'conv2', 'local3', 'local4', 'softmax_linear']:
            with tf.variable_scope(scope, reuse=True):

                w = tf.get_variable('weights')
                b = tf.get_variable('biases')

                w_assign_op = w.assign(data_dict[scope]['weights'])
                b_assign_op = b.assign(data_dict[scope]['biases'])

                sess.run([w_assign_op, b_assign_op])

        step = 0
        start_time = time.time()
        while step < FINE_TUNE_STEPS:

          _, loss_value = sess.run([train_op, loss])

          if step % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - start_time
            start_time = current_time

            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

          step += 1

        if MODEL_SAVE_FILE:
          out = sess.run(['conv1/weights:0',           # (5, 5, 3, 64)
                          'conv1/biases:0',            # (64,)
                          'conv2/weights:0',           # (5, 5, 64, 64)
                          'conv2/biases:0',            # (64,)
                          'local3/weights:0',          # (2304, 384)
                          'local3/biases:0',           # (384,)
                          'local4/weights:0',          # (384, 192)
                          'local4/biases:0',           # (192,)
                          'softmax_linear/weights:0',  # (192, 10)
                          'softmax_linear/biases:0'])  # (10,)

          data_dict = dict()
          data_dict['conv1'] = dict()
          data_dict['conv1']['weights']          = out[0]
          data_dict['conv1']['biases']           = out[1]
          data_dict['conv2'] = dict()
          data_dict['conv2']['weights']          = out[2]
          data_dict['conv2']['biases']           = out[3]
          data_dict['local3'] = dict()
          data_dict['local3']['weights']         = out[4]
          data_dict['local3']['biases']          = out[5]
          data_dict['local4'] = dict()
          data_dict['local4']['weights']         = out[6]
          data_dict['local4']['biases']          = out[7]
          data_dict['softmax_linear'] = dict()
          data_dict['softmax_linear']['weights'] = out[8]
          data_dict['softmax_linear']['biases']  = out[9]

          np.save(MODEL_SAVE_FILE, data_dict)

        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if MODEL_LOAD_FILE is None:
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
  else:
    train_custom()


if __name__ == '__main__':
  tf.app.run()
