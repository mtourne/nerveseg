import tensorflow as tf
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS

import nerveseg_train
import nerveseg_input
import nerveseg
import matplotlib as mpl
import matplotlib.cm
import scipy as scp
import scipy.misc


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/nerveseg',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/nerveseg_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('num_examples', 1,
                            """Number of examples to run.""")

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def color_image(image, num_classes=2):
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def pred_once(saver, summary_writer, pred, summary_op):
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
      while step < num_iter and not coord.should_stop():
          step += 1
          predictions = sess.run([pred])
          predictions = np.array(predictions[0])
          # squeeze depth dimension, to be only height x width
          print(predictions.shape)
          predictions = np.squeeze(predictions, axis=(0,))
          print(predictions.shape)
          colored = color_image(predictions)
          print(colored.shape)
          scp.misc.imsave('output.png', colored)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)




def pred():
    with tf.Graph().as_default() as g:
        images, labels = nerveseg_input.inputs()

        logits, pred = nerveseg.inference(images)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        pred_once(saver, summary_writer, pred, summary_op)


def main(argv=None):  # pylint: disable=unused-argument
    pred()

if __name__ == '__main__':
  tf.app.run()
