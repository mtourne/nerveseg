import tensorflow as tf
import math
import re
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

TOWER_NAME = 'tower'

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def add_conv_relu(bottom_layer, features, name):
  ''' add a convolution '''
  in_features = bottom_layer.get_shape()[3].value
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, in_features, features],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(bottom_layer, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [features], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv)
    return conv


def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[1]
    f = math.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def add_deconv(bottom_layer, features, name):
  with tf.variable_scope(name) as scope:
    k_size = 3
    in_features = bottom_layer.get_shape()[3].value
    # filter has its out_features, in_features reversed compared to
    # kernel for conv2d
    f_shape = [ k_size, k_size, features, in_features ]
    weights = get_deconv_filter(f_shape)
    # Compute shape out of Bottom
    stride = 2
    in_shape = tf.shape(bottom_layer)
    h = ((in_shape[1] - 1) * stride) + 1
    w = ((in_shape[2] - 1) * stride) + 1
    new_shape = [in_shape[0], h, w, features]
    deconv = tf.nn.conv2d_transpose(bottom_layer, weights, new_shape,
                                    [1, stride, stride, 1], padding='SAME')
    _activation_summary(deconv)
    return deconv



def inference(images):
  conv1_1 = add_conv_relu(images, 64, 'conv1_1')
  conv1_2 = add_conv_relu(conv1_1, 64, 'conv1_2')

  pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  out1 = norm1

  conv2_1 = add_conv_relu(out1, 128, 'conv2_1')
  conv2_2 = add_conv_relu(conv2_1, 128, 'conv2_2')

  # norm before pool or vice versa ?
  pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  out2 = norm2

  conv3_1 = add_conv_relu(out2, 256, 'conv3_1')
  conv3_2 = add_conv_relu(conv3_1, 265, 'conv3_2')

  pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
  out3 = norm3

  conv4_1 = add_conv_relu(out3, 512, 'conv4_1')
  conv4_2 = add_conv_relu(conv4_1, 512, 'conv4_2')

  pool4 = tf.nn.max_pool(conv4_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')

  norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm4')

  out4 = norm4

  conv5_1 = add_conv_relu(out4, 1024, 'conv5_1')
  conv5_2 = add_conv_relu(conv5_1, 1024, 'conv5_2')

  out5 = conv5_2

  up6 = add_deconv(out5, 512, 'up6')
  merge6 = tf.concat(1, [up6, out4], name='merge6')
  conv6_1 = add_conv_relu(merge6, 512, 'conv6_1')
  conv6_2 = add_conv_relu(conv6_1, 512, 'conv6_2')

  out6 = conv6_2

  up7 = add_deconv(out6, 256, 'up7')
  merge7 = tf.concat(1, [up7, out3], name='merge7')
  conv7_1 = add_conv_relu(merge7, 256, 'conv7_1')
  conv7_2 = add_conv_relu(conv7_1, 256, 'conv7_2')

  out7 = conv7_2

  up8 = add_deconv(out7, 128, 'up8')
  merge8 = tf.concat(1, [up8, out2], name='merge8')
  conv8_1 = add_conv_relu(merge8, 128, 'conv8_1')
  conv8_2 = add_conv_relu(conv8_1, 128, 'conv8_2')

  out8 = conv8_2

  up9 = add_deconv(out8, 64, 'up9')
  merge9 = tf.concat(1, [up9, out1], name='merge9')
  conv9_1 = add_conv_relu(merge9, 64, 'conv9_1')
  conv9_2 = add_conv_relu(conv9_1, 64, 'conv9_2')

  out9 = conv9_2

  num_classes = 2
  with tf.variable_scope('conv10') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, num_classes],
                                           stddev=5e-2,
                                           wd=0.0)
      conv10 = tf.nn.conv2d(out9, kernel, [1, 1, 1, 1], padding='SAME')
      _activation_summary(conv10)

  out10 = conv10

  logits = out10
  pred = tf.argmax(out10, dimension=3)
  return logits, pred

def loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
      logits
      logits = tf.reshape(logits, (-1, num_classes))
      epsilon = tf.constant(value=1e-4)
      logits = logits + epsilon
      labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

      softmax = tf.nn.softmax(logits)

      if head is not None:
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                              head), reduction_indices=[1])
      else:
        cross_entropy = -tf.reduce_sum(
          labels * tf.log(softmax), reduction_indices=[1])

      cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                          name='xentropy_mean')
      tf.add_to_collection('losses', cross_entropy_mean)

      loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
