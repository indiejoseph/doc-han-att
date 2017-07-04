from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, GRUCell
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from ran_cell import RANCell


L2_REG = 1e-4


class Model(object):
  def __init__(self, conf):
    self.batch_size = conf.batch_size
    self.vocab_size = conf.vocab_size
    self.rnn_size = conf.rnn_size
    self.document_size = conf.document_size
    self.sentence_size = conf.sentence_size
    self.word_attention_size = conf.word_attention_size
    self.sent_attention_size = conf.sent_attention_size
    self.char_embedding_size = conf.char_embedding_size
    self.keep_prob = conf.keep_prob

    self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    self.inputs = tf.placeholder(shape=(self.batch_size, self.document_size, self.sentence_size), dtype=tf.int64, name='inputs')
    self.labels = tf.placeholder(shape=(self.batch_size,), dtype=tf.int64, name='labels')
    self.sentence_lengths = tf.placeholder(shape=(self.batch_size, self.document_size), dtype=tf.int64, name='sentence_lengths')
    self.document_lengths = tf.placeholder(shape=(self.batch_size), dtype=tf.int64, name='document_lengths')

    with tf.device('/cpu:0'):
      self.embedding = tf.get_variable('embedding',
                                       [self.vocab_size, self.char_embedding_size],
                                       trainable=False)
      inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)

    char_length = tf.reshape(self.sentence_lengths, [-1]) # [batch_size * document_size]
    char_inputs = tf.reshape(inputs, [self.batch_size * self.document_size, self.sentence_size, self.char_embedding_size])

    with tf.variable_scope('character_encoder') as scope:
      char_outputs, _ = self.bi_gru_encode(char_inputs, char_length, scope)

      with tf.variable_scope('attention') as scope:
        char_attn_outputs = self.attention(char_outputs, self.word_attention_size, scope)
        char_attn_outputs = tf.reshape(char_attn_outputs, [self.batch_size, self.document_size, -1])

      with tf.variable_scope('dropout'):
        char_attn_outputs = layers.dropout(char_attn_outputs,
                                           keep_prob=self.keep_prob,
                                           is_training=self.is_training)

    with tf.variable_scope('sentence_encoder') as scope:
      sent_outputs, _ = self.bi_gru_encode(char_attn_outputs, self.document_lengths, scope)

      with tf.variable_scope('attention') as scope:
        sent_attn_outputs = self.attention(sent_outputs, self.sent_attention_size, scope)

      with tf.variable_scope('dropout'):
        sent_attn_outputs = layers.dropout(sent_attn_outputs,
                                           keep_prob=self.keep_prob,
                                           is_training=self.is_training)

    with tf.variable_scope('losses'):
      logits = layers.fully_connected(inputs=sent_attn_outputs,
                                      num_outputs=2,
                                      activation_fn=None,
                                      weights_regularizer=layers.l2_regularizer(scale=L2_REG))
      pred = tf.argmax(logits, 1)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=self.labels))
      correct_pred = tf.equal(self.labels, pred)
      correct_pred = tf.cast(correct_pred, tf.float32)
      self.accuracy = tf.reduce_mean(correct_pred)

      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.cost = tf.add_n([loss] + reg_losses)


  def attention(self, inputs, size, scope):
    with tf.variable_scope(scope or 'attention') as scope:
      attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                 shape=[size],
                                                 regularizer=layers.l2_regularizer(scale=L2_REG),
                                                 dtype=tf.float32)
      input_projection = layers.fully_connected(inputs, size,
                                                activation_fn=tf.tanh,
                                                weights_regularizer=layers.l2_regularizer(scale=L2_REG))
      vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
      attention_weights = tf.nn.softmax(vector_attn, dim=1)
      weighted_projection = tf.multiply(inputs, attention_weights)
      outputs = tf.reduce_sum(weighted_projection, axis=1)

    return outputs


  def bi_gru_encode(self, inputs, sentence_size, scope=None):
    batch_size = inputs.get_shape()[0]

    with tf.variable_scope(scope or 'bi_gru_encode'):
      fw_cell = RANCell(self.rnn_size, keep_prob=self.keep_prob, normalize=True, is_training=self.is_training)
      bw_cell = RANCell(self.rnn_size, keep_prob=self.keep_prob, normalize=True, is_training=self.is_training)
      fw_cell_state = fw_cell.zero_state(batch_size, tf.float32)
      bw_cell_state = bw_cell.zero_state(batch_size, tf.float32)

      enc_out, (enc_state_fw, enc_state_bw) = bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                        cell_bw=bw_cell,
                                                                        inputs=inputs,
                                                                        sequence_length=sentence_size,
                                                                        initial_state_fw=fw_cell_state,
                                                                        initial_state_bw=bw_cell_state)

      enc_state = tf.concat([enc_state_fw, enc_state_bw], 1)
      enc_outputs = tf.concat(enc_out, 2)

    return enc_outputs, enc_state


if __name__ == '__main__':
  tf.flags.DEFINE_integer('batch_size', 32, 'Batch Size')
  tf.flags.DEFINE_integer('vocab_size', 1000, 'Vocabulary size')
  tf.flags.DEFINE_integer('word_attention_size', 300, 'Word level attention unit size')
  tf.flags.DEFINE_integer('sent_attention_size', 300, 'Sentence level attention unit size')
  tf.flags.DEFINE_integer('document_size', 16, 'Document size')
  tf.flags.DEFINE_integer('sentence_size', 25, 'Sentence size')
  tf.flags.DEFINE_integer('attention_size', 300, 'Sentence size')
  tf.flags.DEFINE_integer('rnn_size', 300, 'RNN unit size')
  tf.flags.DEFINE_integer('char_embedding_size', 300, 'Embedding dimension')
  tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep prob')
  tf.flags.DEFINE_bool('is_training', True, 'training model')

  FLAGS = tf.flags.FLAGS
  FLAGS._parse_flags()

  model = Model(FLAGS)
