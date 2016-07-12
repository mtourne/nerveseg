import tensorflow as tf
import skimage
import skimage.io
import matplotlib as mpl
import matplotlib.cm
import scipy as scp
import scipy.misc

import nerveseg_pred

tf.app.flags.DEFINE_string('mask_input', '/home/mtourne/data/nerveseg/train/png/labels/10_100_mask.png',
                           """Input image path.""")

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    img1 = skimage.io.imread(FLAGS.mask_input)
    img_rows = 64
    img_cols = 80
    img1 = scp.misc.imresize(img1, (img_rows, img_cols), interp='bicubic')

    colored = nerveseg_pred.color_image(img1)
    scp.misc.imsave('output.png', colored)


if __name__ == '__main__':
  tf.app.run()
