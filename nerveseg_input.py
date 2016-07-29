import tensorflow as tf

import os
import glob
import re
from PIL import Image
import numpy as np
import io
import random
import math

NUM_CLASSES=2
NUMBER_OF_EPOCHES=10000
# roughly 5000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('image_dir', '/home/mtourne/data/nerveseg/train/png',
                           """image directory (png format).""")

FLAGS = tf.app.flags.FLAGS

IMG_WIDTH = 580
IMG_HEIGHT = 420

img_mean = 99.4930907001
img_std = 56.6518606624

def read_labeled_image_dir(data_dir, img_extension='.png', label_extension='_mask.png'):
    ''' read tif files
    use preprocessed mask (label) files
    (white value 255 replaced with 1)
    '''
    #label_dir = os.path.join(data_dir, 'labels')
    glob_pattern = os.path.join(data_dir, '*' + img_extension)
    files = glob.glob(glob_pattern)
    img_names = []
    for img_file in files:
        img_filename = os.path.basename(img_file)
        (img_name, ext) = os.path.splitext(img_filename)
        if re.search('mask', img_file):
            continue
        img_names.append(img_name)
    train_dict = {}
    for img_name in img_names:
        img_file = os.path.join(data_dir, '{}{}'.format(img_name, img_extension))
        label_file = os.path.join(data_dir, '{}{}'.format(img_name, label_extension))
        train_dict[img_name] = (img_file, label_file)
    return train_dict


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
    img_tensor = tf.image.decode_png(img_data, channels=1)
    img_tensor = tf.reshape(img_tensor, [IMG_HEIGHT, IMG_WIDTH, 1])
    # transform to a float image
    img_tensor = tf.cast(img_tensor, tf.float32)
    # img_tensor = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32, name=None)


    label_data = tf.read_file(label_filename, name='read_label')
    label_tensor = tf.image.decode_png(label_data, channels=1)
    label_tensor = tf.reshape(label_tensor, [IMG_HEIGHT, IMG_WIDTH, 1])
    label_tensor = tf.cast(label_tensor, tf.float32)
    # label_tensor = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32, name=None)
    return img_tensor, label_tensor

def preprocess_image(image):
    #image = tf.image.per_image_whitening(image)
    image = tf.sub(image, img_mean) # mean for data centering
    image = tf.div(image, img_std)
    return image

def preprocess_label(label):
    label = tf.div(label, 255.0) # scale 0 to 1
    return label

def image_distortions(image, distortions):
    distort_left_right_random = distortions[0]
    mirror = tf.less(tf.pack([1.0, distort_left_right_random, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    distort_up_down_random = distortions[1]
    mirror = tf.less(tf.pack([distort_up_down_random, 1.0, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    return image

def get_train_cross_val_set(train_dict):
    # try to get images from all the patients
    cross_val_images = []
    cross_val_labels = []
    train_images = []
    train_labels = []
    image_names = train_dict.keys()
    image_per_patient = {}
    for image_name in image_names:
        patient, image_number = image_name.split('_')
        patient_images = image_per_patient.get(patient)
        if not patient_images:
            patient_images = {
                'count': 0,
                'images': [],
            }
            image_per_patient[patient] = patient_images
        patient_images['count'] += 1
        patient_images['images'].append(image_name)
    patients = len(image_per_patient.keys())
    print("patients total: {}".format(patients))
    for patient, patient_images in image_per_patient.iteritems():
        count_per_patient = patient_images['count']
        pick = int(count_per_patient * 0.20)
        picks = random.sample(patient_images['images'], pick)
        for pick in picks:
            (image_file, label_file) = train_dict.pop(pick)
            cross_val_images.append(image_file)
            cross_val_labels.append(label_file)
    # iterate over what's left in train dict and put them into
    # train  lists
    for image_name, files in train_dict.iteritems():
        (image_file, label_file) = files
        train_images.append(image_file)
        train_labels.append(label_file)
    return train_images, train_labels, cross_val_images, cross_val_labels


def inputs():
    train_dict = read_labeled_image_dir(FLAGS.image_dir)

    train_images, train_labels, cross_val_images, cross_val_labels = (
        get_train_cross_val_set(train_dict))
    train_count = len(train_images)
    xval_count = len(cross_val_images)

    images = tf.convert_to_tensor(train_images, dtype=tf.string)
    labels = tf.convert_to_tensor(train_labels, dtype=tf.string)

    xval_images = tf.convert_to_tensor(cross_val_images, dtype=tf.string)
    xval_labels = tf.convert_to_tensor(cross_val_labels, dtype=tf.string)

    # Makes an input queue
    # not setting num_epochs, it can cycle through the slices an
    # unlimited number of times
    input_queue = tf.train.slice_input_producer([images, labels],
                                                shuffle=True)

    image, label = read_images_from_disk(input_queue)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image = preprocess_image(image)
    label = preprocess_label(label)

    distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)

    image = image_distortions(image, distortions)
    label = image_distortions(label, distortions)

    num_preprocess_threads = 16
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=FLAGS.batch_size,
                                              num_threads=num_preprocess_threads)

    # create small patches
    #image_batch = tf.extract_image_patches(image_batch, 'SAME',
    #                                       ksizes=[1, 32, 32, 1],
    #                                       strides=[1, 32, 32, 1],
    #                                       rates=[1, 32, 32, 1])
    #label_batch = tf.extract_image_patches(label_batch, 'SAME',
    #                                       ksizes=[1, 32, 32, 1],
    #                                       strides=[1, 32, 32, 1],
    #                                       rates=[1, 32, 32, 1])

    ## process xval
    xval_images = tf.convert_to_tensor(cross_val_images, dtype=tf.string)
    xval_labels = tf.convert_to_tensor(cross_val_labels, dtype=tf.string)
    xval_input_queue = tf.train.slice_input_producer([xval_images, xval_labels])
    xval_image, xval_label = read_images_from_disk(xval_input_queue)
    xval_image = preprocess_image(xval_image)
    xval_label = preprocess_label(xval_label)
    xval_image_batch, xval_label_batch = tf.train.batch([xval_image, xval_label],
                                                        batch_size=FLAGS.batch_size,
                                                        num_threads=num_preprocess_threads)

    # resize images
    # XX (mtourne): why would that work better than the full size image with more information
    #img_rows = 64
    #img_cols = 80
    #image_batch = tf.image.resize_bicubic(image_batch, [img_rows, img_cols], name='resize_image')
    #label_batch = tf.image.resize_bicubic(label_batch, [img_rows, img_cols], name='resize_label')
    #
    #xval_image_batch = tf.image.resize_bicubic(xval_image_batch, [img_rows, img_cols],
    #                                           name='resize_xval_image')
    #xval_label_batch = tf.image.resize_bicubic(xval_label_batch, [img_rows, img_cols],
    #                                           name='resize_xval_label')

    tf.image_summary('images', image_batch, max_images = 5)

    return {
        'image': image_batch,
        'label': label_batch,
        'image_count': train_count,
        'xval_image': xval_image_batch,
        'xval_label': xval_label_batch,
        'xval_count': xval_count,
    }
