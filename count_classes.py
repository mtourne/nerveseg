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
    for name, (_, label_filename) in train_dict.iteritems():
        img = skimage.io.imread(label_filename)
        arr = np.array(img)
        unique, counts = np.unique(arr, return_counts=True)
        for i, value in enumerate(unique):
            if value == 0:
                zero_count += counts[i]
            elif value == 255:
                one_count += counts[i]
            else:
                print("element with value: {} found".format(value))
                exit()
        if n % 20 == 0:
            print("There are {} backround and {} foreground".format(zero_count, one_count))
        n += 1
    print("There are {} backround and {} foreground".format(zero_count, one_count))

if __name__ == '__main__':
  tf.app.run()
