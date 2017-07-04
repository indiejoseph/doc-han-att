import collections
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs


_checked_scope = core_rnn_cell_impl._checked_scope
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def orthogonal(shape):
  """Orthogonal initilaizer."""
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)


def orthogonal_initializer(scale=1.0):
  """Orthogonal initializer."""
  def _initializer(shape, dtype=tf.float32, partition_info=None):  # pylint: disable=unused-argument
    return tf.constant(orthogonal(shape) * scale, dtype)

  return _initializer


def linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None,
           kernel_regularizer=None,
           bias_regularizer=None,
           normalize=False):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
    kernel_regularizer: kernel regularizer
    bias_regularizer: bias regularizer
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer,
        regularizer=kernel_regularizer)

    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)

    if normalize:
      res = tf.contrib.layers.layer_norm(res)

    # remove the layer’s bias if there is one (because it would be redundant)
    if not bias or normalize:
      return res

    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer,
          regularizer=bias_regularizer)

  return nn_ops.bias_add(res, biases)


class RANCell(RNNCell):
  """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

  def __init__(self, num_units, input_size=None, activation=tanh, keep_prob=0.5,
               normalize=False, reuse=None, is_training=tf.constant(False)):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._normalize = normalize
    self._keep_prob = keep_prob
    self._reuse = reuse
    self._is_training = is_training

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._num_units, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "ran_cell", reuse=self._reuse):
      with vs.variable_scope("gates"):
        c, h = state
        gates = tf.nn.sigmoid(linear([inputs, h], 2 * self._num_units, True,
                                     normalize=self._normalize,
                                     kernel_initializer=tf.orthogonal_initializer()))
        i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)

      with vs.variable_scope("candidate"):
        content = linear([inputs], self._num_units, True, normalize=self._normalize)

      new_c = i * content + f * c
      new_h = self._activation(c)

      new_h = tf.cond(self._is_training,
                      lambda: nn_ops.dropout(new_h, self._keep_prob),
                      lambda: new_h)

      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      output = new_h
    return output, new_state
