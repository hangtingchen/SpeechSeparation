# -*- coding: utf-8 -*-
'''
@data: 2018/10/31
@note: The code is derived from lstm in tf
        LSTMCell2 add bias in projection compared with LSTMCell
        LSTMCell2P add non-recurrent option compared with LSTMCell2
'''

from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.python.keras import initializers
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.keras import activations
import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class LSTMPCell(LayerRNNCell):
  """
  Follow LSTMCell, but add bias for projection
  add non-recurrent and recurrent part of lstm
  """

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, num_recurent_proj=None,
               proj_clip=None, num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, init_dict=None,
               reuse=None, name=None, dtype=None, **kwargs):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`. It
        could also be string that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().

      When restoring from CudnnLSTM-trained checkpoints, use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(LSTMPCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer) if initializer else initializers.get(xavier_initializer())
    self._num_proj = num_proj
    self._num_recurent_proj=num_recurent_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._init_dict=init_dict
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_recurent_proj is None else self._num_recurent_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units],
        initializer=tf.constant_initializer(self._init_dict[_WEIGHTS_VARIABLE_NAME]) if isinstance(self._init_dict,dict) else self._initializer,
        partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=tf.constant_initializer(self._init_dict[_BIAS_VARIABLE_NAME]) if isinstance(self._init_dict,dict) else self._initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                         initializer=tf.constant_initializer(self._init_dict["_w_f_diag"]) if isinstance(self._init_dict,dict) else self._initializer)
      self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                         initializer=tf.constant_initializer(self._init_dict["_w_i_diag"]) if isinstance(self._init_dict,dict) else self._initializer)
      self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                         initializer=tf.constant_initializer(self._init_dict["_w_o_diag"]) if isinstance(self._init_dict,dict) else self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection.%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=tf.constant_initializer(self._init_dict["projection.%s" % _WEIGHTS_VARIABLE_NAME]) if isinstance(self._init_dict,dict) else self._initializer,
          partitioner=maybe_proj_partitioner)
      self._proj_bias=self.add_variable(
          "projection.%s" % _BIAS_VARIABLE_NAME,
          shape=[self._num_proj],
          initializer=tf.constant_initializer(self._init_dict["projection.%s" % _BIAS_VARIABLE_NAME]) if isinstance(self._init_dict,dict) else self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, `[batch, num_units].
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev_total) = state
      m_prev=array_ops.slice(m_prev_total,[0,0],[-1,self._num_recurent_proj])
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, self._num_recurent_proj])

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    # Diagonal connections
    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = nn_ops.bias_add(math_ops.matmul(m, self._proj_kernel),self._proj_bias)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(LSTMPCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

