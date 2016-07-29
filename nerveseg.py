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

# weight decay
#WD=1e-6
#WD=0.0
#WD=5e-4
WD=None

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1e-2  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-3      # Initial learning rate.

TOWER_NAME = 'tower'

background = 1356132911.0
foreground = 16553089.0
ratio = foreground / (background + foreground)
print("Class ratio: {}".format(ratio))
class_weight = [ratio, 1.0 - ratio]


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

def _variable_with_weight_decay(name, shape, stddev=None, wd=None):
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
    #tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=dtype)
  )
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


def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def add_conv_relu(bottom_layer, features, name,
                  phase_train,
                  do_batch_norm=True,
                  keep_prob=None,
                  in_features=None, k_shape=None,
                  k_size=3, stride=1):
  ''' add a convolution '''
  if not in_features:
    # calculate from the bottom layer
    in_features = bottom_layer.get_shape()[3].value
  stddev = 0.1
  if not k_shape:
    k_shape = [k_size, k_size, in_features, features]
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=k_shape,
                                         stddev=stddev,
                                         wd=WD)
    conv = tf.nn.conv2d(bottom_layer, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [features], tf.constant_initializer(0.1))
    conv = tf.nn.bias_add(conv, biases)
    if do_batch_norm:
      conv = batch_norm(conv, features, phase_train)
    conv = tf.nn.relu(conv, name=scope.name)
    if FLAGS.debug:
      conv = tf.Print(conv, [tf.shape(conv)],
                        message='Shape of {}'.format(name),
                        summarize=4, first_n=1)
    _activation_summary(conv)
    if keep_prob is not None:
      conv = tf.nn.dropout(conv, keep_prob)
    return conv


def get_deconv_filter1(f_shape):
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

def get_deconv_filter2(f_shape):
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

    filt = tf.zeros_initializer(shape=weights.shape, dtype=tf.float32)
    filt = filt + weights
    return filt

def get_deconv_filter(f_shape):
  return get_deconv_filter2(f_shape)

def add_deconv(bottom_layer, features, name, match_layer_shape=None, k_size=3):
  stride = 2

  if match_layer_shape is None:
    in_shape = tf.shape(bottom_layer)
    # No shape provided
    # Compute shape out of bottom layer
    #h = ((in_shape[1] - 1) * stride) + 1
    #w = ((in_shape[2] - 1) * stride) + 1
    h = in_shape[1] * 2
    w = in_shape[2] * 2
  else:
    # match layer shape
    in_shape = tf.shape(match_layer_shape)
    h = in_shape[1]
    w = in_shape[2]

  with tf.variable_scope(name) as scope:
    in_features = bottom_layer.get_shape()[3].value
    # filter has its out_features, in_features reversed compared to
    # kernel for conv2d
    f_shape = [ k_size, k_size, features, in_features ]
    # weights is bilinear filter from FCN
    #kernel = get_deconv_filter(f_shape)
    # try with xavier like in u-net release
    kernel = _variable_with_weight_decay('weights',
                                         shape=f_shape,
                                         wd=WD)

    new_shape = [in_shape[0], h, w, features]
    output_shape = tf.pack(new_shape)
    deconv = tf.nn.conv2d_transpose(bottom_layer, kernel, output_shape,
                                    [1, stride, stride, 1], padding='SAME')
    # add biases and relu
    biases = _variable_on_cpu('biases', [features], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(deconv, biases)
    deconv = tf.nn.relu(deconv, name=scope.name)
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

def inference(images, keep_prob, phase_train):
  use_max_pool = True
  conv1_1 = add_conv_relu(images, 64, 'conv1_1', phase_train,
                          k_size=3, keep_prob=keep_prob)
  conv1_2 = add_conv_relu(conv1_1, 64, 'conv1_2', phase_train,
                          k_size=3, keep_prob=keep_prob)
  if use_max_pool:
    out1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool1')

  else:
    ## replace max pool with a conv 3x3 with stride 2
    out1 = add_conv_relu(conv1_2, 64, 'conv_pool1', phase_train,
                         do_batch_norm=False,
                         k_size=3, stride=2)

  conv2_1 = add_conv_relu(out1, 128, 'conv2_1', phase_train,
                          k_size=3, keep_prob=keep_prob)
  conv2_2 = add_conv_relu(conv2_1, 128, 'conv2_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  if use_max_pool:
    out2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool2')
  else:
    out2 = add_conv_relu(conv2_2, 128, 'conv_pool2', phase_train,
                         do_batch_norm=False,
                         k_size=3, stride=2)

  conv3_1 = add_conv_relu(out2, 256, 'conv3_1', phase_train,
                          k_size=3, keep_prob=keep_prob)
  conv3_2 = add_conv_relu(conv3_1, 256, 'conv3_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  if use_max_pool:
    out3 = tf.nn.max_pool(conv3_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool3')
  else:
    out3 = add_conv_relu(conv3_2, 256, 'conv_pool3', phase_train,
                         do_batch_norm=False,
                         k_size=3, stride=2)

  conv4_1 = add_conv_relu(out3, 512, 'conv4_1', phase_train,
                          k_size=3, keep_prob=keep_prob)
  conv4_2 = add_conv_relu(conv4_1, 512, 'conv4_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  if use_max_pool:
    out4 = tf.nn.max_pool(conv4_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool3')
  else:
    out4 =  add_conv_relu(conv4_2, 512, 'conv_pool4', phase_train,
                          do_batch_norm=False,
                          k_size=3, stride=2)

  conv5_1 = add_conv_relu(out4, 1024, 'conv5_1', phase_train,
                          k_size=3, keep_prob=keep_prob)
  conv5_2 = add_conv_relu(conv5_1, 1024, 'conv5_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  out5 = conv5_2

  up6 = add_deconv(out5, 512, 'up6', k_size=2
                   #, match_layer_shape=conv4_2
  )
  # get output of conv4 pre max_pool
  merge6 = add_merge([up6, conv4_2], 'merge6')
  # specify the number of features coming from merge6
  # (somehow merge6 dim3 is not fully defined)
  conv6_1 = add_conv_relu(merge6, 512, 'conv6_1', phase_train,
                          k_size=3, in_features=1024
                          , keep_prob=keep_prob)
  conv6_2 = add_conv_relu(conv6_1, 512, 'conv6_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  out6 = conv6_2

  up7 = add_deconv(out6, 256, 'up7', k_size=2
                   #, match_layer_shape=conv3_2
  )
  # get output of conv3 pre max_pool
  merge7 = add_merge([up7, conv3_2], 'merge7')
  conv7_1 = add_conv_relu(merge7, 256, 'conv7_1', phase_train,
                          k_size=3, in_features=512
                          , keep_prob=keep_prob)
  conv7_2 = add_conv_relu(conv7_1, 256, 'conv7_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  out7 = conv7_2

  up8 = add_deconv(out7, 128, 'up8', k_size=2
                   #, match_layer_shape=conv2_2
  )
  merge8 = add_merge([up8, conv2_2], 'merge8')
  conv8_1 = add_conv_relu(merge8, 128, 'conv8_1', phase_train,
                          k_size=3, in_features=256
                          , keep_prob=keep_prob)
  conv8_2 = add_conv_relu(conv8_1, 128, 'conv8_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  out8 = conv8_2

  up9 = add_deconv(out8, 64, 'up9', k_size=2
                   #, match_layer_shape=conv1_2
  )
  merge9 = add_merge([up9, conv1_2], 'merge9')
  conv9_1 = add_conv_relu(merge9, 64, 'conv9_1', phase_train,
                          k_size=3, in_features=128
                          , keep_prob=keep_prob)
  conv9_2 = add_conv_relu(conv9_1, 64, 'conv9_2', phase_train,
                          k_size=3, keep_prob=keep_prob)

  out9 = conv9_2

  num_classes = 2
  with tf.variable_scope('conv10') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 64, num_classes],
                                         stddev=0.1,
                                         wd=WD)
    conv10 = tf.nn.conv2d(conv9_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.1))
    conv10 = tf.nn.bias_add(conv10, biases)

  logits = conv10

  # version 1 - same as v3
  #splits = tf.split(0, FLAGS.batch_size, logits)
  #preds = []
  #output_maps = []
  #for split in splits:
  #  split = tf.squeeze(split)
  #  shape = split.get_shape()
  #  split = tf.reshape(split, [-1, num_classes])
  #  pixel_softmax = tf.nn.softmax(split)
  #  pred_split = tf.argmax(pixel_softmax, dimension=1)
  #  pred_split = tf.reshape(pred_split, [shape[0].value, shape[1].value])
  #  pixel_softmax = tf.reshape(pixel_softmax, shape)
  #  output_maps.append(pixel_softmax)
  #  preds.append(pred_split)
  #
  #pred = tf.pack(preds)
  #output_map2 = tf.pack(output_maps)

  # version 2
  output_map = pixel_wise_softmax(logits)
  pred = tf.argmax(output_map, dimension=3)

  net_parts = {
    'pool1': out1,
    'pool2': out2,
    'pool3': out3,
    'pool4': out4,
    'last_downconv': out5,

    'deconv1': up6,
    'deconv2': up7,
    'deconv3': up8,
    'deconv4': up9,

    #'output_map2': output_map2,
  }

  return output_map, pred, net_parts

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence)

def loss(logits, labels, num_classes):
  # put a bigger loss factor on class 1 (the mask!)
  # XX (mtourne): why is it called 'head' ?
  return loss_fcn(logits, labels, num_classes, head=class_weight)
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


# require reshaping labels so it's [batch_size, num_classes], which
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
      ## labels is a vector
      labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
      #
      softmax = tf.nn.softmax(logits)

      labels = tf.one_hot(labels, num_classes, on_value=1.0, off_value=0.0,
                          dtype=tf.float32)

      if head is not None:
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                              head), reduction_indices=[1])
      else:
        cross_entropy = -tf.reduce_sum(
          labels * tf.log(softmax), reduction_indices=[1])

      #cross_entropy = -tf.reduce_sum(
      #   labels * tf.log(softmax), reduction_indices=[1])


      #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
      #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits, labels, 0.7,
      #                                                         name='pos_weighted_cross')
      cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                          name='cross_entropy_mean')
      tf.add_to_collection('losses', cross_entropy_mean)
      #cross_entropy_sum = tf.reduce_sum(cross_entropy,
      #                                  name='cross_entropy_sum')
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
