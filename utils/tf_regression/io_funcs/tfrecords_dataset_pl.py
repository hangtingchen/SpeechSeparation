#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

"""Utility functions for import data using tf.contrib.data.Dataset.
Make sure TensorFlow vesion >= 1.2.0
"""

from __future__ import absolute_import
from __future__ import print_function
import datetime
import tensorflow as tf

def get_batch(filename,batch_size,input_dim,\
                output_dim,left_context=0,right_context=0,\
                num_epochs=1,infer=False,num_threads=8,\
                transform_fn=None):
    def splice_feats(feats, left_context, right_context):
        """Splice feats like KALDI.
        Args:
            feats: input feats have a shape [row, col].
            left: left context number.
            right: right context number.
        Returns:
            Spliced feats with a shape [row, col*(left+1+right)]
        """
        sfeats = []
        row = tf.shape(feats)[0]
        # Left
        ##这个splice的意思
        for i in range(left_context, 0, -1):
            fl = tf.slice(feats, [0, 0], [row-i, -1])
            for j in range(i):
                fl = tf.pad(fl, [[1, 0], [0, 0]], mode='SYMMETRIC')
            sfeats.append(fl)
        sfeats.append(feats)

        # Right
        for i in range(1, right_context+1):
            fr = tf.slice(feats, [i, 0], [-1, -1])
            for j in range(i):
                fr = tf.pad(fr, [[0, 1], [0, 0]], mode='SYMMETRIC')
            sfeats.append(fr)
        return tf.concat(sfeats, 1)

    def parser_train(record):
        """Extract data from a `tf.SequenceExamples` protocol buffer for training."""
        context_features = {
            'name': tf.FixedLenFeature([], tf.string),
        }
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_dim],dtype=tf.float32),
            'outputs': tf.FixedLenSequenceFeature(shape=[output_dim],dtype=tf.float32),
            'noises': tf.FixedLenSequenceFeature(shape=[output_dim],dtype=tf.float32),
            'outputs_pl': tf.FixedLenSequenceFeature(shape=[output_dim],dtype=tf.float32),
            'noises_pl': tf.FixedLenSequenceFeature(shape=[output_dim],dtype=tf.float32),
        }

        context, sequence = tf.parse_single_sequence_example(
            record,
            context_features=context_features,
            sequence_features=sequence_features)
        if(transform_fn is None):
            inp=sequence['inputs']
            outp=sequence['outputs']
            outp_pl=sequence['outputs_pl']
        else:
            inp,outp,outp_pl=transform_fn(sequence['inputs'],sequence['outputs'],sequence['noises'],sequence['outputs_pl'],sequence['noises_pl'])
        splice_inputs = splice_feats(inp, left_context, right_context)
        return context['name'], splice_inputs, outp, outp_pl
    
    def parser_infer(record):
        """Extract data from a `tf.SequenceExamples` protocol buffer for training."""
        context_features = {
            'name': tf.FixedLenFeature([], tf.string),
        }
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_dim],
                                                dtype=tf.float32),
            "inputs_ang": tf.FixedLenSequenceFeature(shape=[input_dim],
                                                dtype=tf.float32)
        }

        context, sequence = tf.parse_single_sequence_example(
            record,
            context_features=context_features,
            sequence_features=sequence_features)
        if(transform_fn is None):
            inp=sequence['inputs']
        else:
            inp,_,_=transform_fn(sequence['inputs'],None,None,None,None)
        splice_inputs = splice_feats(inp, left_context, right_context)
        return context['name'], splice_inputs, sequence['inputs_ang']

    buffer_size = 10 * batch_size
    tf.logging.info("Buffer size is {}".format(buffer_size))
    dataset = tf.data.TFRecordDataset(filename)
    if(not infer):
        dataset=dataset.map(parser_train,num_parallel_calls=num_threads)
        dataset=dataset.map(lambda name,inputs,outputs,outputs_pl:\
                (name,inputs,outputs,outputs_pl,tf.shape(inputs)[0]),num_parallel_calls=num_threads)
        dataset = dataset.prefetch(buffer_size)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape([None,input_dim*(left_context + 1 + right_context)]),
                tf.TensorShape([None,output_dim]),
                tf.TensorShape([None,output_dim]),
                tf.TensorShape([]),
            ),
            padding_values=(
                '',
                0.0,
                0.0,
                0.0,
                0
            )
        )
        dataset_iter=dataset.make_one_shot_iterator()
        name,inputs,outputs,outputs_pl,lengths=dataset_iter.get_next()
        return name,inputs,outputs,outputs_pl,lengths
    else:
        assert num_epochs==1 and batch_size==1, "Unexpeted epoch {} and batch size {}".format(num_epochs,batch_size)
        dataset=dataset.map(parser_infer,num_parallel_calls=num_threads)
        dataset=dataset.map(lambda name,inputs,inputs_angle:\
                (name,inputs,inputs_angle,tf.shape(inputs)[0]),num_parallel_calls=num_threads)
        dataset.prefetch(buffer_size)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(1)
        dataset_iter=dataset.make_one_shot_iterator()
        name,inputs,inputs_angle,lengths=dataset_iter.get_next()
        return name,inputs,inputs_angle,lengths

def get_num_samples(filename,batch_size,input_dim,output_dim,infer=False):
    """ Get number of bacthes. """
    counter = 0

    with tf.Graph().as_default():
        if(infer):
            _, inputs,_,_ = get_batch(filename,
                batch_size,input_dim,output_dim,infer=infer)
        else:
            _, inputs,_,_,_ = get_batch(filename,
                batch_size,input_dim,output_dim,infer=infer)

        init = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)
        start = datetime.datetime.now()
    while True:
        try:
            inputs_np=sess.run(inputs)
            # print(inputs_np)
            counter += inputs_np.shape[0]
        except tf.errors.OutOfRangeError:
            end = datetime.datetime.now()
            duration = (end - start).total_seconds()
            tf.logging.info('Number of samples is %d. Reading time is %.2fs.' % (
                counter, duration))
            break
    sess.close()
    return counter