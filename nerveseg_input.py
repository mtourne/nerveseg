import tensorflow as tf

import os
import glob
import re
from PIL import Image
import numpy as np
import io

#BATCH_SIZE=10
NUMBER_OF_EPOCHES=10000

tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a batch.""")

FLAGS = tf.app.flags.FLAGS

IMG_WIDTH = 580
IMG_HEIGHT = 420


def read_labeled_image_dir(data_dir):
    ''' read tif files
    use preprocessed mask (label) files
    (white value 255 replaced with 1)
    '''
    img_extension = '.png'
    label_extension = '_mask.png'
    label_dir = os.path.join(data_dir, 'labels')
    glob_pattern = os.path.join(data_dir, '*' + img_extension)
    files = glob.glob(glob_pattern)
    img_names = []
    for img_file in files:
        img_filename = os.path.basename(img_file)
        (img_name, ext) = os.path.splitext(img_filename)
        if re.search('mask', img_file):
            continue
        img_names.append(img_name)
    img_files = []
    label_files = []
    for img_name in img_names:
        img_files.append(os.path.join(data_dir, '{}{}'.format(img_name, img_extension)))
        label_files.append(os.path.join(label_dir, '{}{}'.format(img_name, label_extension)))


    return img_files, label_files


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filenames for data and tensor
    Returns:
      Two tensors: the decoded image, and the string label.
    """

    img_filename = input_queue[0]
    label_filename = input_queue[1]

    img_data = tf.read_file(img_filename, name='read_image')
    #img_tensor = tf.image.decode_png(img_data, channels=1)
    #img_tensor = tf.reshape(img_tensor, [IMG_HEIGHT, IMG_WIDTH, 1])
    # transform to a float image
    # img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32, name=None)


    label_data = tf.read_file(label_filename, name='read_label')
    #label_tensor = tf.image.decode_png(label_data, channels=1)
    # label_tensor = tf.reshape(label_tensor, [IMG_HEIGHT, IMG_WIDTH, 1])
    label_tensor = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32, name=None)
    return img_tensor, label_tensor

def preprocess_image(image):
    image = tf.image.per_image_whitening(image)
    return image

def inputs():
    # XX (mtourne): make configurable
    image_list, label_list = read_labeled_image_dir('/home/mtourne/data/nerveseg/train/png')

    print(image_list[0])
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.string)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=NUMBER_OF_EPOCHES,
                                                shuffle=True)

    image, label = read_images_from_disk(input_queue)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image = preprocess_image(image)
    #label = preprocess_label(label)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=FLAGS.batch_size)
    tf.image_summary('images', image_batch)

    return image_batch, label_batch
