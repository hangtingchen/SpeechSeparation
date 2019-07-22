# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/01/21
@note : A network for progressive learning
'''

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, fully_connected, batch_norm

sys.path.append(os.path.dirname(sys.path[0]))
from models.base import Model
from models.LSTMPCell import LSTMPCell

class ProgressiveLearningTrainer(Model):
    def __init__(self,sess,args,cross_validation=False, infer=False,
                name='ProgressiveLearningTrainer'):
        super(ProgressiveLearningTrainer,self).__init__(sess,name)
        self.max_grad_norm = 15
        self.batch_size = args.batch_size
        # self.l2_scale = args.l2_scale
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.batch_size = args.batch_size
        self.cross_validation = cross_validation
        self.littleK=0.1;self.bigK=1.0
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size, None, \
            self.input_dim * (self.left_context + 1 + self.right_context)], \
            name=self.name+"/data/inputs")
        self.pl_labels = tf.placeholder(
            tf.float32, [self.batch_size, None, self.input_dim], \
            name=self.name+"/data/pl_labels")
        self.labels = tf.placeholder(
            tf.float32, [self.batch_size, None, self.output_dim], \
            name=self.name+"/data/labels")
        self.lengths = tf.placeholder(tf.int32, \
            [self.batch_size ], name="/data/lengths")

        if args.net_type.lower() == 'blstm':
            raise ValueError("Unexpected net type {}".format(args.net_type))
        elif args.net_type.lower() == 'dnn':
            self.net_output=self.build_DNN(args)
        else:
            raise ValueError("Unexpected net type {}".format(args.net_type))
        if not infer:
            self.build_ops(args)
        # else:
        #     self.net_output=tf.reduce_mean([self.net_output,*self.hidden_outputs[1:]],axis=0)

    def build_DNN(self,args,reuse=False):
        self.num_hidden_layers=args.num_hidden_layers
        self.cell_dim=args.cell_dim
        self.hidden_outputs=[]
        tf.logging.info("DNN is {}*{}".format(self.num_hidden_layers,self.cell_dim))

        relu_stddev = np.sqrt(2.0 / self.cell_dim)
        relu_itializer = tf.truncated_normal_initializer(mean=0.0, stddev=relu_stddev)
        normalizer_params = {
            "is_training": False if self.cross_validation else True,
            "scale": True,
            "renorm": True,
            "decay": 0.99,}
        
        with tf.variable_scope(self.name+"/net") as scope:
            h=self.inputs;self.hidden_outputs.append(h)
            for layer in range(self.num_hidden_layers):
                h = tf.concat(self.hidden_outputs,2)
                h = fully_connected(h, self.cell_dim,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=normalizer_params,
                    weights_initializer=relu_itializer,
                    weights_regularizer=None,
                    biases_initializer=tf.zeros_initializer())
                h = fully_connected(h, self.input_dim,
                    activation_fn=tf.nn.sigmoid,
                    weights_initializer=xavier_initializer(),
                    biases_initializer=tf.zeros_initializer())
                self.hidden_outputs.append(h)
            h = tf.concat(self.hidden_outputs,2)
            y = fully_connected(h, self.output_dim,
                activation_fn=None,
                weights_initializer=xavier_initializer(),
                biases_initializer=tf.zeros_initializer())
            y = self.post_process(y,args.mapping_mode)
            return y

    def build_ops(self,args):
        variable_averages = tf.train.ExponentialMovingAverage(
                            self.MOVING_AVERAGE_DECAY,self.NUM_UPDATES)
        with tf.variable_scope(tf.get_variable_scope()):
            self.init_lr = args.lr
            self.global_step=tf.Variable(args.start_step,trainable=False)
            self.lr = tf.train.exponential_decay(self.init_lr, self.global_step,
                int(args.train_num_samples/args.batch_size), 0.5, staircase=True)
            
            max_length=tf.reduce_max(self.lengths)
            masks = tf.sequence_mask(lengths=self.lengths, \
                    maxlen=max_length, dtype=tf.bool)
            loss=self.bigK*tf.squared_difference(self.net_output,self.labels)
            if not self.cross_validation:
                for l in range(1,self.num_hidden_layers+1):
                    loss+=self.littleK*tf.squared_difference(self.hidden_outputs[l],self.pl_labels)
            loss= tf.boolean_mask(loss,masks)
            # self.debughook1=tf.reduce_min(tf.reshape(self.labels,[-1]))
            self.loss = tf.reduce_mean(tf.reshape(loss,[-1]))
            if not self.cross_validation:
                self.loss_ratio = tf.sqrt(self.loss/tf.reduce_mean(tf.reshape(self.bigK*tf.pow(self.labels,2.0)+self.littleK*self.num_hidden_layers*tf.pow(self.pl_labels,2.0),[-1])))
            else:
                self.loss_ratio = tf.sqrt(self.loss/tf.reduce_mean(tf.reshape(self.bigK*tf.pow(self.labels,2.0),[-1])))
            adam=tf.train.AdamOptimizer(self.lr)
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads = adam.compute_gradients(self.loss)
                for i,(g,v) in enumerate(grads):
                    grads[i]=(tf.clip_by_norm(g,self.max_grad_norm),v)
                apply_gradient_op=adam.apply_gradients(grads,global_step=self.global_step)
                variables_averages_op=variable_averages.apply()
            self.opts=tf.group(apply_gradient_op,variables_averages_op)

    def post_process(self,y,mapping_mode):
        if(mapping_mode=='DM'):
            # return y
            raise ValueError("Unexpected mode for target {}".format(mapping_mode))
        elif(mapping_mode=='IRM'):
            return tf.sigmoid(y)
        elif(mapping_mode=='IM'):
            # return tf.log(tf.sigmoid(y))
            raise ValueError("Unexpected mode for target {}".format(mapping_mode))
        else:
            raise ValueError("Unexpected mode for target {}".format(mapping_mode))

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
    row = tf.shape(feats)[1]
    # Left
    ##这个splice的意思
    for i in range(left_context, 0, -1):
        fl = tf.slice(feats, [0, 0, 0], [-1, row-i, -1])
        for j in range(i):
            fl = tf.pad(fl, [[0, 0], [1, 0], [0, 0]], mode='SYMMETRIC')
        sfeats.append(fl)
    sfeats.append(feats)

    # Right
    for i in range(1, right_context+1):
        fr = tf.slice(feats, [0, i, 0], [-1, -1, -1])
        for j in range(i):
            fr = tf.pad(fr, [[0, 0], [0, 1], [0, 0]], mode='SYMMETRIC')
        sfeats.append(fr)
    return tf.concat(sfeats, 2)
