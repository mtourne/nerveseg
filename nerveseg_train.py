import tensorflow as tf
import numpy as np
from datetime import datetime
import os.path
import time

import nerveseg
import nerveseg_input
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/nerveseg',
                           """Directory where to write event logs """
                           """and checkpoint.""")
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
  # flatten
  labels = tf.reshape(labels, [-1])
  pred = tf.reshape(pred, [-1])
  intersection = tf.reduce_sum(tf.mul(pred, labels))
  return ((2. * intersection + smooth) /
          (tf.reduce_sum(labels) + tf.reduce_sum(pred)  + smooth))


def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    images, labels = nerveseg_input.inputs()

    logits, pred = nerveseg.inference(images)

    loss = nerveseg.loss(logits, labels, 2)

    # get dice coeff
    coeff = dice_coeff(pred, labels)

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

    if FLAGS.resume:
      print('Attempting resuming from: {}'.format(FLAGS.train_dir))
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
      else:
        print('No checkpoint file found')
        return

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        coeff_value = sess.run([coeff])
        format_str = ('%s: step %d, loss = %.4f, dice_coeff = %.4f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             coeff_value[0],
                             examples_per_sec, sec_per_batch))

      # 100 originally
      if step % 10 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
  if not FLAGS.resume:
    print("(Re)creating dir: {}".format(FLAGS.train_dir))
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
