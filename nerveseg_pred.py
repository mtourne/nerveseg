import tensorflow as tf
import numpy as np
import math
import skimage
import skimage.io
import skimage.transform

FLAGS = tf.app.flags.FLAGS

import nerveseg_train
import nerveseg_input
import nerveseg
import matplotlib as mpl
import matplotlib.cm
import scipy as scp
import scipy.misc

tf.app.flags.DEFINE_string('input', '/home/mtourne/data/nerveseg/train/png/10_100.png',
                           """Input image path.""")

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/nerveseg',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/nerveseg_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('num_examples', 1,
                            """Number of examples to run.""")

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def color_image(image, max_class=1):
    #print(image)
    norm = mpl.colors.Normalize(vmin=0., vmax=max_class)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def pred_once(saver, summary_writer, pred, summary_op, feed_dict):
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

    predictions = sess.run([pred], feed_dict=feed_dict)
    predictions = np.array(predictions[0])
    # squeeze depth dimension, to be only height x width
    predictions = np.squeeze(predictions, axis=(0,))
    colored = color_image(predictions)
    print("Writing output.png")
    scp.misc.imsave('output.png', colored)

def pred():
    img1 = skimage.io.imread(FLAGS.input)
    img1 = np.expand_dims(img1, axis=2)

    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=
                                (nerveseg_input.IMG_HEIGHT,
                                 nerveseg_input.IMG_WIDTH, 1))
        feed_dict = {images: img1}

        # preprocess input image the same as training data.
        images = nerveseg_input.preprocess_image(images)

        batch_images = tf.expand_dims(images, 0)
        # resize image to mimic nerveseg_input.py
        #img_rows = 64
        #img_cols = 80
        #batch_images  = tf.image.resize_bicubic(batch_images, [img_rows, img_cols], name='resize_image')
        logits, pred = nerveseg.inference(batch_images)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        pred_once(saver, summary_writer, pred, summary_op, feed_dict)


def main(argv=None):  # pylint: disable=unused-argument
    pred()

if __name__ == '__main__':
  tf.app.run()
