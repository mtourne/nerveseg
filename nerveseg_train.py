import tensorflow as tf
import numpy as np
from datetime import datetime
import os.path
import time
import math
import tdb

import nerveseg
import nerveseg_input
import time

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('train_dir', '/tmp/nerveseg',
#                           """Directory where to write event logs """
#                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('resume', True,
                            """ resume training """)

smooth = 1.

def dice_coeff(pred, labels):
  labels = tf.cast(labels, tf.float32)
  pred = tf.cast(pred, tf.float32)
  # flatten, keep batches
  labels = tf.reshape(labels, [FLAGS.batch_size, -1])
  pred = tf.reshape(pred, [FLAGS.batch_size, -1])
  intersection = tf.reduce_sum(tf.mul(pred, labels), reduction_indices=[1])
  return ((2. * intersection + smooth) /
          (tf.reduce_sum(labels, reduction_indices=[1]) +
           tf.reduce_sum(pred, reduction_indices=[1])  + smooth))

def label_max(labels):
  # reduce accross col, width, depth, keep batch
  label_max = tf.reduce_max(labels, reduction_indices=[1,2,3])
  return label_max

def test_xval(sess, coeff_placeholder, input_dict):
  print("Test XVAL")
  coeff_values = []
  images = input_dict['image']
  labels = input_dict['label']
  xval_images = input_dict['xval_image']
  xval_labels = input_dict['xval_label']
  xval_count = input_dict['xval_count']
  num_examples_per_step = FLAGS.batch_size
  step = 0
  num_iter = int(math.ceil(xval_count / FLAGS.batch_size))
  all_coeff_values = []
  while step < num_iter:
    step += 1
    image_data, label_data = sess.run([xval_images, xval_labels])
    coeff_values = sess.run([coeff_placeholder], feed_dict={images: image_data, labels: label_data})
    all_coeff_values.extend(coeff_values)
  avg_coeff = np.average(all_coeff_values)
  print("Average dice coeff value: {}".format(avg_coeff))

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    input_dict = nerveseg_input.inputs()
    images = input_dict['image']
    labels = input_dict['label']

    logits, pred = nerveseg.inference(images, train=True)

    loss = nerveseg.loss(logits, labels, 2)
    coeff_placeholder = dice_coeff(pred, labels)
    label_max_placeholder = label_max(labels)

    train_op = nerveseg.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # possibly resume training
    if FLAGS.resume:
      print('Attempting resuming from: {}'.format(FLAGS.train_dir))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    try:
      step = 0
      coeffs = []
      losses = []
      masks_present = 0
      while not coord.should_stop():
        start_time = time.time()
        _, loss_value, coeff_values, label_max_values = (
          sess.run([train_op, loss, coeff_placeholder, label_max_placeholder]))
        #status, result = tdb.debug([train_op, loss, coeff_placeholder, label_max_placeholder],
        #                           session=sess)
        #loss_value = result[1]
        #coeff_values = result[2]
        #label_max_values = result[3]
        duration = time.time() - start_time
        for i, label_max_value in enumerate(label_max_values):
          if label_max_value > 0:
            coeffs.append(coeff_values[i])
            masks_present += 1

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        losses.append(loss_value)

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          avg_coeffs = np.average(coeffs)
          avg_loss = np.average(losses)
          format_str = ('%s: step %d, loss = %.4f, avg_dice_coeff = %.4f, masks present = %d, avg_loss = %.4f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               avg_coeffs, masks_present, avg_loss,
                               examples_per_sec, sec_per_batch))
          coeffs = []
          losses = []
          masks_present = 0

        if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          test_xval(sess, coeff_placeholder, input_dict)
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

        step += 1

    finally:
      coord.request_stop()

      coord.join(threads)
      sess.close()


def main(argv=None):
  if not FLAGS.resume:
    print("(Re)creating dir: {}".format(FLAGS.train_dir))
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
