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

"""Feature extraction for network compression for CIFAR-10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import argparse

import numpy as np
import tensorflow as tf

import cifar10
from DataManager import save_data_to_file

def parse_args():

    parser = argparse.ArgumentParser(description='CIFAR-10 feature extraction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--conv1-channels',
                        dest    = 'conv1_channels',
                        help    = 'number of channels in first convolutional layer',
                        default = 64,
                        type    = int)

    parser.add_argument('--samples-per_category',
                        dest    = 'samples_per_category',
                        help    = 'number of samples to take in count per category',
                        default = 500,
                        type    = int)

    parser.add_argument('--output-file',
                        dest    = 'output_file',
                        help    = 'output file for extracted compression data',
                        default = './data/cifar10_features.npz',
                        type    = str)

    args = parser.parse_args()

    return args

# Parse script arguments
args = parse_args()

OUTPUT_FILE = args.output_file
SAMPLES_PER_CATEGORY = args.samples_per_category

NUM_CONV1_CHANNELS = args.conv1_channels
if NUM_CONV1_CHANNELS == 64:
    MODEL_DIR = r'./data/models/baseline'
else:
    MODEL_DIR = os.path.join(r'./data/models/', NUM_CONV1_CHANNELS)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './data/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', MODEL_DIR,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, labels_op,
              conv2_in_op, conv2_out_op, summary_op):
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
      conv2_in_list = []
      conv2_out_list = [] 
      cnt_arr = np.zeros((10,), dtype=np.int16)
      while step < num_iter and not coord.should_stop():
        predictions, labels, conv2_in, conv2_out = sess.run([top_k_op, labels_op, conv2_in_op, conv2_out_op])
        true_count += np.sum(predictions)
        cnt_arr[labels] += 1
        if cnt_arr[labels] <= SAMPLES_PER_CATEGORY:
            conv2_in_list.append(conv2_in)
            conv2_out_list.append(conv2_out)
        if step % 100 == 0:
            print('%s: step %d' % (datetime.now(), step))
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)

      # Extract and save data needed for Thinet compression
      conv1_w, conv2_w = sess.run(['conv1/weights:0', 'conv2/weights:0'])
      conv2_in_arr = np.squeeze(np.array(conv2_in_list))
      conv2_out_arr = np.squeeze(np.array(conv2_out_list))
      save_data_to_file(OUTPUT_FILE, conv1_w, conv2_w, conv2_in_arr, conv2_out_arr)
      print('Saved %d samples to %s: ' % (conv2_in_arr.shape[0], OUTPUT_FILE))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data = eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, conv2_in, conv2_out = cifar10.inference(images, extract_features = True)

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
      eval_once(saver, summary_writer, top_k_op, labels, conv2_in, conv2_out, summary_op)
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
