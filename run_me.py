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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

# Supress TensorFLow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")


def eval_once(top_k_op, k, model_file, msg):
  """Run Eval once.

  Args:
    top_k_op:   Top K op.
    model_file: Model file in npy format.
  """
  with tf.Session() as sess:

    data_dict = np.load(model_file, encoding='latin1').item()

    for scope in ['conv1', 'conv2', 'local3', 'local4', 'softmax_linear']:
        with tf.variable_scope(scope, reuse=True):

            w = tf.get_variable('weights')
            b = tf.get_variable('biases')

            w_assign_op = w.assign(data_dict[scope]['weights'])
            b_assign_op = b.assign(data_dict[scope]['biases'])

            sess.run([w_assign_op, b_assign_op])

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ k.
      precision = 100 * (true_count / total_sample_count)
      print('%s : accuracy top-%d = %.3f [%%]' % (msg, k, precision))

    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():

  for conv1_channels in [64, 48, 32, 24, 16, 8]:

    for top_k in [1, 5]:

      ########################
      #       Base line      #
      ########################

      model_file = r'./data/models_npy/model_%d.npy' % conv1_channels
      log_msg = r'%d conv1 channels, trained from scratch' % (conv1_channels)

      with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = cifar10.inputs(eval_data=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images, conv1_channels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, top_k)

        eval_once(top_k_op, top_k, model_file, log_msg)

      time.sleep(1)

      if conv1_channels == 64:
        continue

      ########################
      # Experimental results #
      ########################

      # Without reconstruction & without fine tuning
      model_file = r'./data/models_without_reconst/model_compressed_%d.npy' % conv1_channels
      log_msg = r'%d conv1 channels, w/o reconstruction, w/o fine tuning' % (conv1_channels)

      with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = cifar10.inputs(eval_data=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images, conv1_channels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, top_k)

        eval_once(top_k_op, top_k, model_file, log_msg)

      time.sleep(1)

      # With reconstruction & without fine tuning
      model_file = r'./data/models_with_reconst/model_compressed_%d.npy' % conv1_channels
      log_msg = r'%d conv1 channels, with reconstruction, w/o fine tuning' % (conv1_channels)

      with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = cifar10.inputs(eval_data=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images, conv1_channels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, top_k)

        eval_once(top_k_op, top_k, model_file, log_msg)

      time.sleep(1)

      # Without reconstruction & with fine tuning
      model_file = r'./data/models_fine_tuned/fine_tunned_without_reconst_%d.npy' % conv1_channels
      log_msg = r'%d conv1 channels, w/o reconstruction, with fine tuning' % (conv1_channels)

      with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = cifar10.inputs(eval_data=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images, conv1_channels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, top_k)

        eval_once(top_k_op, top_k, model_file, log_msg)

      time.sleep(1)

      # With reconstruction & with fine tuning
      model_file = r'./data/models_fine_tuned/fine_tunned_with_reconst_%d.npy' % conv1_channels
      log_msg = r'%d conv1 channels, with reconstruction, with fine tuning' % (conv1_channels)

      with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = cifar10.inputs(eval_data=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images, conv1_channels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, top_k)

        eval_once(top_k_op, top_k, model_file, log_msg)

      time.sleep(1)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  evaluate()


if __name__ == '__main__':
  tf.app.run()
