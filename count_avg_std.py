import tensorflow as tf
import nerveseg_input
import skimage
import skimage.io
import numpy as np

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    train_dict = nerveseg_input.read_labeled_image_dir(FLAGS.image_dir)
    zero_count = 0
    one_count = 0
    n = 0
    imgs = []
    total = len(train_dict.keys())
    for name, (image_filename, _) in train_dict.iteritems():
        img = skimage.io.imread(image_filename)
        imgs.append(img)
        if n % 100 == 0:
            print('Done: {0}/{1} images'.format(n, total))
        if n % 1000 == 0:
            mean = np.mean(imgs)
            std = np.std(imgs)
            print("Mean: {}, Std: {}".format(mean, std))
        n += 1
    mean = np.mean(imgs)
    std = np.std(imgs)
    print("Mean: {}, Std: {}".format(mean, std))

if __name__ == '__main__':
  tf.app.run()
