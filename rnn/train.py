"""
Created on Wed Apr 25 15:13:46 2018

@author: WP
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary = read_data(FLAGS.text)
print('*************Data size', len(vocabulary))


with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')


model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build(embedding_file='embedding.npy')


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')

    for x in range(2):
        logging.debug('epoch [{0}]....'.format(x))
        state = sess.run(model.state_tensor)
        dls = utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
        for dl in dls:
            ##################
            # Your Code here
            ##################
            input_x = dl[0]
            input_y = dl[1]
            feed_dict = {model.X:input_x, model.Y:input_y, model.keep_prob:0.8, model.state_tensor:state}           

            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % 10 == 0:
                logging.debug('step [{0}] loss [{1}] total step [{2}]'.format(gs, l, dls.shape[0]))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
