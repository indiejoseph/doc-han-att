from __future__ import print_function
import os
import pprint
import time
import tensorflow as tf
import numpy as np
from model import Model
from data_helpers import get_vocab

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'epochs')
flags.DEFINE_integer('rnn_size', 300, 'RNN unit size')
flags.DEFINE_integer('word_attention_size', 300, 'Word level attention unit size')
flags.DEFINE_integer('sent_attention_size', 300, 'Sentence level attention unit size')
flags.DEFINE_integer('char_embedding_size', 300, 'Embedding dimension')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'Directory name to save the checkpoints [checkpoint]')
flags.DEFINE_integer('vocab_size', 6790, 'vocabulary size')
flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep prob')
flags.DEFINE_integer('document_size', 30, 'document size')
flags.DEFINE_integer('sentence_size', 50, 'sentence size')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_float('grad_clip', 5.0, 'grad clip')

FLAGS = flags.FLAGS

def read_records(index=0):
  train_queue = tf.train.string_input_producer(['./data/train.tfrecords'], num_epochs=FLAGS.epochs)
  valid_queue = tf.train.string_input_producer(['./data/valid.tfrecords'], num_epochs=FLAGS.epochs)
  queue = tf.QueueBase.from_list(index, [train_queue, valid_queue])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'sentence_lengths': tf.FixedLenFeature([FLAGS.document_size], tf.int64),
          'document_lengths': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64),
          'text': tf.FixedLenFeature([FLAGS.document_size * FLAGS.sentence_size], tf.int64),
      })

  sentence_lengths = features['sentence_lengths']
  document_lengths = features['document_lengths']
  label = features['label']
  text = features['text']

  sentence_lengths_batch, document_lengths_batch, label_batch, text_batch = tf.train.shuffle_batch(
      [sentence_lengths, document_lengths, label, text],
      batch_size=FLAGS.batch_size,
      capacity=5000,
      min_after_dequeue=1000)

  return sentence_lengths_batch, document_lengths_batch, label_batch, text_batch


def main(_):
  pp.pprint(FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print(' [*] Creating checkpoint directory...')
    os.makedirs(FLAGS.checkpoint_dir)

  checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')

  # load pre-trained char embedding
  char_emb = np.load('./data/emb.npy')

  sentence_lengths_batch, document_lengths_batch, label_batch, text_batch = read_records()
  valid_sentence_lengths_batch, valid_document_lengths_batch, valid_label_batch, valid_text_batch = read_records(1)

  text_batch = tf.reshape(text_batch, (-1, FLAGS.document_size, FLAGS.sentence_size))
  valid_text_batch = tf.reshape(valid_text_batch, (-1, FLAGS.document_size, FLAGS.sentence_size))

  with tf.variable_scope('model'):
    train_model = Model(FLAGS)
  with tf.variable_scope('model', reuse=True):
    valid_model = Model(FLAGS)

  # training operator
  global_step = tf.Variable(0, name='global_step', trainable=False)
  lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 10000, 0.9)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(train_model.cost, tvars), FLAGS.grad_clip)
  optimizer = tf.train.AdamOptimizer(lr)
  train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

  tf.summary.scalar('train_loss', train_model.cost)
  tf.summary.scalar('valid_loss', valid_model.cost)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    # assign char embedding
    sess.run([], feed_dict={train_model.embedding: char_emb})
    sess.run([], feed_dict={valid_model.embedding: char_emb})

    # saver.restore(sess, checkpoint_path)

    # stock_emb = train_model.label_embedding.eval()
    #
    # np.save('./data/stock_emb.npy', stock_emb)
    # print('done')

    # summary_op = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter('./log/train', sess.graph)
    # valid_writer = tf.summary.FileWriter('./log/test')

    current_step = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    valid_cost = 0
    valid_accuracy = 0
    train_cost = 0
    VALID_SIZE = 54

    _, chars = get_vocab()

    try:
      while not coord.should_stop():
        start = time.time()

        if current_step % 500 == 0:
          valid_cost = 0
          for _ in range(VALID_SIZE):
            valid_text, valid_label, valid_sentence_lengths, valid_document_lengths =\
              sess.run([valid_text_batch, valid_label_batch, valid_sentence_lengths_batch, valid_document_lengths_batch])

            valid_outputs = sess.run([valid_model.cost, valid_model.accuracy], feed_dict={
                valid_model.inputs: valid_text,
                valid_model.labels: valid_label,
                valid_model.sentence_lengths: valid_sentence_lengths,
                valid_model.document_lengths: valid_document_lengths,
                valid_model.is_training: False
            })
            valid_cost += valid_outputs[0]
            valid_accuracy += valid_outputs[1]
          valid_cost /= VALID_SIZE
          valid_accuracy /= VALID_SIZE

        inputs, labels, sentence_lengths, document_lengths =\
          sess.run([text_batch, label_batch, sentence_lengths_batch, document_lengths_batch])

        # valid_writer.add_summary(summary, current_step)
        train_cost, train_accuracy, _ = sess.run([train_model.cost, train_model.accuracy, train_op], feed_dict={
            train_model.inputs: inputs,
            train_model.labels: labels,
            train_model.sentence_lengths: sentence_lengths,
            train_model.document_lengths: document_lengths,
            train_model.is_training: True
        })
        # train_writer.add_summary(summary, current_step)
        end = time.time()

        print('Cost at step %s: %s(%s), test cost: %s(%s), time: %s' %
              (current_step, train_cost, train_accuracy, valid_cost, valid_accuracy, end - start))

        current_step = tf.train.global_step(sess, global_step)

        if current_step != 0 and current_step % 1000 == 0:
          save_path = saver.save(sess, checkpoint_path)
          print('Model saved in file:', save_path)

    except tf.errors.OutOfRangeError:
      print('Done training!')
    finally:
      coord.request_stop()

    save_path = saver.save(sess, checkpoint_path)
    print('Model saved in file:', save_path)

    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
