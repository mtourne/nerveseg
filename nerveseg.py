import tensorflow as tf
import math
import re
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
# XX (mtourne): print op seem to messs up the graph
# is it just visualization or also compute ?
tf.app.flags.DEFINE_boolean('debug', False,
                            """Debug messages.""")

import nerveseg_input

NUM_CLASSES = nerveseg_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = nerveseg_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = nerveseg_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

TOWER_NAME = 'tower'

def _print_shape(tensor, name):
  if FLAGS.debug:
    tensor = tf.Print(tensor, [tf.shape(tensor)],
                      message='Shape of {}'.format(name),
                      summarize=4, first_n=1)
  return tensor

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

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


def add_conv_relu(bottom_layer, features, name, in_features=None):
  ''' add a convolution '''
  if not in_features:
    # calculate from the bottom layer
    in_features = bottom_layer.get_shape()[3].value
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, in_features, features],
                                         stddev=5e-2,
                                         wd=0.0)
    #kernel = _variable_with_weight_decay('weights',
    #                                     shape=[3, 3, in_features, features],
    #                                     stddev=5e-2,
    #                                     wd=None)
    conv = tf.nn.conv2d(bottom_layer, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [features], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(bias, name=scope.name)
    if FLAGS.debug:
      conv = tf.Print(conv, [tf.shape(conv)],
                        message='Shape of {}'.format(name),
                        summarize=4, first_n=1)
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


def add_deconv(bottom_layer, features, name, match_layer_shape=None):
  with tf.variable_scope(name) as scope:
    k_size = 3
    stride = 2
    in_features = bottom_layer.get_shape()[3].value
    # filter has its out_features, in_features reversed compared to
    # kernel for conv2d
    f_shape = [ k_size, k_size, features, in_features ]
    weights = get_deconv_filter(f_shape)

    if match_layer_shape is None:
      in_shape = tf.shape(bottom_layer)
      # No shape provided
      # Compute shape out of bottom layer
      h = ((in_shape[1] - 1) * stride) + 1
      w = ((in_shape[2] - 1) * stride) + 1
    else:
      # match layer shape
      in_shape = tf.shape(match_layer_shape)
      h = in_shape[1]
      w = in_shape[2]
    new_shape = [in_shape[0], h, w, features]
    output_shape = tf.pack(new_shape)
    deconv = tf.nn.conv2d_transpose(bottom_layer, weights, output_shape,
                                    [1, stride, stride, 1], padding='SAME')
    if FLAGS.debug:
      deconv = tf.Print(deconv, [tf.shape(deconv)],
                        message='Shape of {}'.format(name),
                        summarize=4, first_n=1)
    _activation_summary(deconv)
    return deconv

def add_merge(layers_to_merge, name):
  merge = tf.concat(3, layers_to_merge, name=name)
  if FLAGS.debug:
    merge = _print_shape(merge, name)
  return merge

def inference(images):
  conv1_1 = add_conv_relu(images, 64, 'conv1_1')
  conv1_2 = add_conv_relu(conv1_1, 64, 'conv1_2')

  pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  #out1 = norm1
  out1 = pool1

  conv2_1 = add_conv_relu(out1, 128, 'conv2_1')
  conv2_2 = add_conv_relu(conv2_1, 128, 'conv2_2')

  # norm before pool or vice versa ?
  pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  #out2 = norm2
  out2 = pool2

  conv3_1 = add_conv_relu(out2, 256, 'conv3_1')
  conv3_2 = add_conv_relu(conv3_1, 256, 'conv3_2')

  pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
  #out3 = norm3
  out3 = pool3

  conv4_1 = add_conv_relu(out3, 512, 'conv4_1')
  conv4_2 = add_conv_relu(conv4_1, 512, 'conv4_2')

  pool4 = tf.nn.max_pool(conv4_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')

  norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm4')

  #out4 = norm4
  out4 = pool4

  conv5_1 = add_conv_relu(out4, 1024, 'conv5_1')
  conv5_2 = add_conv_relu(conv5_1, 1024, 'conv5_2')

  out5 = conv5_2

  up6 = add_deconv(out5, 512, 'up6', match_layer_shape=conv4_2)
  # get output of conv4 pre max_pool
  merge6 = add_merge([up6, conv4_2], 'merge6')
  # specify the number of features coming from merge6
  # (somehow merge6 dim3 is not fully defined)
  conv6_1 = add_conv_relu(merge6, 512, 'conv6_1', in_features=1024)
  conv6_2 = add_conv_relu(conv6_1, 512, 'conv6_2')

  out6 = conv6_2

  up7 = add_deconv(out6, 256, 'up7', match_layer_shape=conv3_2)
  # get output of conv3 pre max_pool
  merge7 = add_merge([up7, conv3_2], 'merge7')
  conv7_1 = add_conv_relu(merge7, 256, 'conv7_1', in_features=512)
  conv7_2 = add_conv_relu(conv7_1, 256, 'conv7_2')

  out7 = conv7_2

  up8 = add_deconv(out7, 128, 'up8', match_layer_shape=conv2_2)
  merge8 = add_merge([up8, conv2_2], 'merge8')
  conv8_1 = add_conv_relu(merge8, 128, 'conv8_1', in_features=256)
  conv8_2 = add_conv_relu(conv8_1, 128, 'conv8_2')

  out8 = conv8_2

  up9 = add_deconv(out8, 64, 'up9', match_layer_shape=conv1_2)
  merge9 = add_merge([up9, conv1_2], 'merge9')
  conv9_1 = add_conv_relu(merge9, 64, 'conv9_1', in_features=128)
  conv9_2 = add_conv_relu(conv9_1, 64, 'conv9_2')

  out9 = conv9_2

  num_classes = 2
  with tf.variable_scope('conv10') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 64, num_classes],
                                         stddev=5e-2,
                                         wd=0.0)
    conv10 = tf.nn.conv2d(out9, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv10, biases)
    # no ReLu for conv10
    conv10 = bias
    if FLAGS.debug:
      conv10 = tf.Print(conv10, [tf.shape(conv10)],
                        message='Shape of conv10',
                        summarize=4, first_n=1)
    _activation_summary(conv10)

  out10 = conv10

  logits = out10

  pred = tf.argmax(out10, dimension=3)
  return logits, pred

def loss(logits, labels, num_classes):
  return loss_fcn(logits, labels, num_classes)
  #return loss_sparse_softmax(logits, labels, num_classes)


def loss_sparse_softmax(logits, labels, num_classes):
  labels = tf.cast(labels, tf.int32)
  logits = tf.reshape(logits, [-1, num_classes])
  labels = tf.reshape(labels, [-1])
  logits = _print_shape(logits, 'logits')
  labels = _print_shape(labels, 'labels')
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  #cross_entropy_sum = tf.reduce_sum(cross_entropy, name='cross_entropy_sum')
  tf.add_to_collection('losses', cross_entropy_mean)

  # XX (mtourne): not sure about this sentence, investigate.
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


# loss from fcn network (not used)
# would require reshaping labels so it's [batch_size, num_classes], which
# each label[i] = (0,1) of pixel being within class
def loss_fcn(logits, labels, num_classes, head=None):
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
      logits = tf.reshape(logits, (-1, num_classes))
      epsilon = tf.constant(value=1e-4)
      logits = logits + epsilon
      # labels is a vector
      labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
      #labels_len = labels.get_shape()[0].value
      #labels_len = len(labels)
      # sparse to dense so that
      # label[i] = 0 becomes label[i] = [1, 0]
      # labels = tf.sparse_to_dense(labels, [labels_len], 1)
      labels = tf.one_hot(labels, num_classes, on_value=1.0, off_value=0.0)
      #labels = tf.to_float(labels)
      #labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

      softmax = tf.nn.softmax(logits)

      if head is not None:
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                              head), reduction_indices=[1])
      else:
        cross_entropy = -tf.reduce_sum(
          labels * tf.log(softmax), reduction_indices=[1])

      cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                    name='cross_entropy_mean')
      tf.add_to_collection('losses', cross_entropy_mean)
      #cross_entropy_sum = tf.reduce_sum(cross_entropy,
      #                                   name='cross_entropy_sum')
      #tf.add_to_collection('losses', cross_entropy_sum)

      loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def train(loss, global_step):
  return train_adam(loss, global_step)
  #return train_cifar(loss, global_step)

def train_adam(total_loss, global_step):
  ''' from doc: Calling minimize() takes care of both computing the gradients and applying them to the variables. If you want to process the gradients before applying them you can instead use the optimizer in three steps:

  1. Compute the gradients with compute_gradients().
  2. Process the gradients as you wish.
  3. Apply the processed gradients with apply_gradients().
  '''
  ## add from train_cifar
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  ## Adam optimizer
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(1e-6)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')


  return train_op

def train_cifar(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
