import cv2
import numpy as np
import os
# could probably load with cv2
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

import nerveseg_train
import nerveseg_input
import nerveseg_pred
import nerveseg

tf.app.flags.DEFINE_string('input_dir', '/home/mtourne/data/nerveseg/test/',
                           """Input image path.""")
tf.app.flags.DEFINE_boolean('write_masks', False,
                            """Write output mask images""")

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 255., cv2.THRESH_BINARY)[1].astype(np.uint8)
    # cv2.resize is width, height, compared to tf (height, width)
    img = cv2.resize(img, (nerveseg_input.IMG_WIDTH, nerveseg_input.IMG_HEIGHT),
                     interpolation=cv2.INTER_CUBIC)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def write_submission(result):
    first_row = 'img,pixels'
    file_name = 'submission.csv'
    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for key in sorted(result):
            s = str(key) + ',' + result[key]
            f.write(s + '\n')

def pred_dir(saver, summary_writer, pred, summary_op, batch_images_raw):
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

    img_files, _ = nerveseg_input.read_labeled_image_dir(FLAGS.input_dir, img_extension='.tif')
    total_files = len(img_files)
    l = 0
    result = {}
    while len(img_files) > 0:
        img_batch = []
        img_filenames = []
        for i in xrange(FLAGS.batch_size):
            if l % 100 == 0:
                print('{}/{}'.format(l, total_files))
            l += 1
            if len(img_files) == 0:
                # case for the last image
                # add the last loaded image n times to
                # have a full batch
                img_batch.extend([img for x in xrange(i, FLAGS.batch_size)])
                i -= 1
                break
            img_filename = img_files.pop(0)
            img = skimage.io.imread(img_filename)
            # add a 1 channel to the image
            img = np.expand_dims(img, axis=2)
            img_batch.append(img)
            img_filenames.append(img_filename)
        predictions = sess.run([pred], feed_dict={batch_images_raw: img_batch})
        predictions = np.array(predictions)
        j = i
        ## XX (mtourne): not quite sure why the output is
        ## shaped depth, batch, rows, cols (I would think batch, depth, rows, cols)
        # Make it of shape depth, batch, rows, cols
        predictions = np.transpose(predictions, axes=[1, 0, 2, 3])
        for i in xrange(j + 1):
            prediction = predictions[i]
            img_filename = img_filenames[i]
            prediction = np.squeeze(prediction, axis=(0,))
            prediction = prep(prediction)
            # write down prediction
            img_filename = os.path.basename(img_filename)
            (img_name, ext) = os.path.splitext(img_filename)
            if FLAGS.write_masks:
                new_img_file = os.path.join(FLAGS.eval_dir, '{}{}'.format(img_name, '.tif'))
                print("writing file: {}".format(new_img_file))
                cv2.imwrite(new_img_file, prediction)
            encoded_pixels = run_length_enc(prediction)
            # convert to int so that sort later is not legicographic
            result[int(img_name)] = encoded_pixels
    print("Writing results")
    write_submission(result)


def pred():
    img1 = skimage.io.imread(FLAGS.input)
    img1 = np.expand_dims(img1, axis=2)

    with tf.Graph().as_default() as g:
        batch_images_list = []
        for i in xrange(FLAGS.batch_size):
            image = tf.placeholder(tf.float32, shape=
                                    (nerveseg_input.IMG_HEIGHT,
                                     nerveseg_input.IMG_WIDTH, 1))


            # preprocess input image the same as training data.
            image = nerveseg_input.preprocess_image(image)
            batch_images_list.append(image)

        batch_images_raw = tf.pack(batch_images_list)
        # resize image to mimic nerveseg_input.py
        img_rows = 64
        img_cols = 80
        batch_images  = tf.image.resize_bicubic(batch_images_raw,
                                                [img_rows, img_cols], name='resize_image')
        _, pred = nerveseg.inference(batch_images)


        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        pred_dir(saver, summary_writer, pred, summary_op, batch_images_raw)


def main(argv=None):  # pylint: disable=unused-argument
    pred()

if __name__ == '__main__':
  tf.app.run()
