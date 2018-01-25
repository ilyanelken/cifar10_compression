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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import math
import time
import argparse

import numpy as np
import tensorflow as tf

import cifar10

def parse_args():

    parser = argparse.ArgumentParser(description='CIFAR-10 evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--conv1-channels',
                        dest    = 'conv1_channels',
                        help    = 'number of channels in first convolutional layer',
                        default = 64,
                        type    = int)

    parser.add_argument('--save-file',
                        dest    = 'save_file',
                        help    = 'model weights in *.npy file',
                        default = None,
                        type    = str)

    args = parser.parse_args()

    return args

# Parse script arguments
args = parse_args()

MODEL_SAVE_FILE = args.save_file
NUM_CONV1_CHANNELS = args.conv1_channels
if NUM_CONV1_CHANNELS == 64:
    MODEL_DIR = r'./data/models/baseline'
else:
    MODEL_DIR = os.path.join(r'./data/models/', str(NUM_CONV1_CHANNELS))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './data/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', MODEL_DIR,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def get_graph_num_params():

    total_parameters = 0

    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    return total_parameters

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    print("Total number of parameters: %d" % get_graph_num_params())

    if MODEL_SAVE_FILE is not None:
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

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, NUM_CONV1_CHANNELS)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
