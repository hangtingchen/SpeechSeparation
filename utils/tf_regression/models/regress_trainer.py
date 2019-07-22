# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/12/11
@note : A lstmp network for regression tasks
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

class RegressTrainer(Model):
    def __init__(self,sess,args,cross_validation=False, infer=False,
                name='RegressTrainer'):
        super(RegressTrainer,self).__init__(sess,name)
        self.max_grad_norm = 15
        self.batch_size = args.batch_size
        # self.l2_scale = args.l2_scale
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.batch_size = args.batch_size
        self.cross_validation = cross_validation
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size, None, \
            self.input_dim * (self.left_context + 1 + self.right_context)], \
            name=self.name+"/data/inputs")
        self.labels = tf.placeholder(
            tf.float32, [self.batch_size, None, self.output_dim], \
            name=self.name+"/data/labels")
        self.lengths = tf.placeholder(tf.int32, \
            [self.batch_size ], name="/data/lengths")

        if args.net_type.lower() == 'blstm':
            self.net_output=self.build_BLSTM(args)
        elif args.net_type.lower() == 'dnn':
            self.net_output=self.build_DNN(args)
        else:
            raise ValueError("Unexpected net type {}".format(args.net_type))
        if not infer:
            self.build_ops(args)

    def build_DNN(self,args,reuse=False):
        self.num_hidden_layers=args.num_hidden_layers
        self.cell_dim=args.cell_dim
        tf.logging.info("DNN is {}*{}".format(self.num_hidden_layers,self.cell_dim))

        relu_stddev = np.sqrt(2.0 / self.cell_dim)
        relu_itializer = tf.truncated_normal_initializer(mean=0.0, stddev=relu_stddev)
        normalizer_params = {
            "is_training": False if self.cross_validation else True,
            # "is_training": True,
            "scale": True,
            "renorm": True,
            "decay": 0.99,}

        with tf.variable_scope(self.name+"/net") as scope:
            h=self.inputs
            for layer in range(self.num_hidden_layers):
                h = fully_connected(h, self.cell_dim,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=normalizer_params,
                    weights_initializer=relu_itializer,
                    weights_regularizer=None,
                    biases_initializer=tf.zeros_initializer())
                if(layer==0):
                    if(args.mapping_mode=='DM'):h_res=h
                    else:h_res=tf.zeros_like(h)
            y = fully_connected(h+h_res, self.output_dim,
                activation_fn=None,
                weights_initializer=xavier_initializer(),
                biases_initializer=tf.zeros_initializer())
            y = self.post_process(y,args.mapping_mode)
            return y

    def build_BLSTM(self,args,reuse=False):
        self.cell_dim=args.cell_dim
        self.proj_dim=args.proj_dim
        self.recur_dim=args.recur_dim
        self.num_hidden_layers=args.num_hidden_layers
        tf.logging.info("BLSTM is {}*{}-{}-{}".format(self.cell_dim,\
            self.num_hidden_layers,self.proj_dim,self.recur_dim))

        with tf.variable_scope(self.name+"/net") as scope:
            # if reuse:scope.reuse_variables()
            def lstm_cell():
                return LSTMPCell(
                    self.cell_dim, use_peepholes=True,
                    num_proj=self.proj_dim,
                    num_recurent_proj=self.recur_dim,
                    forget_bias=0.0, state_is_tuple=True,
                    activation=tf.tanh,
                    reuse=reuse)
            with tf.variable_scope("lstmp"):
                cell_fw=tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_hidden_layers)], state_is_tuple=True)
                cell_bw=tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_hidden_layers)], state_is_tuple=True)
                initial_states_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                initial_states_bw = cell_bw.zero_state(self.batch_size, tf.float32)
                outputs,_=tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    self.inputs,
                    initial_state_fw=initial_states_fw,
                    initial_state_bw=initial_states_bw,
                    dtype=tf.float32,
                    sequence_length=self.lengths,
                )
                outputs=tf.concat(outputs, 2)
                if(args.mapping_mode=='DM'):
                    h_res = fully_connected(self.inputs, self.proj_dim*2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=xavier_initializer(),
                        biases_initializer=tf.zeros_initializer())
                else:
                    h_res=tf.zeros_like(outputs)

            with tf.variable_scope("out_layer"):
                y = fully_connected(outputs+h_res, self.output_dim,
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
            loss= tf.boolean_mask(tf.squared_difference(self.net_output,self.labels),masks)
            # self.debughook1=tf.reduce_min(tf.reshape(self.labels,[-1]))
            self.loss = tf.reduce_mean(tf.reshape(loss,[-1]))
            self.loss_ratio = tf.sqrt(self.loss/tf.reduce_mean(tf.reshape(tf.pow(self.labels,2.0),[-1])))
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
            return y
        elif(mapping_mode=='IRM'):
            return tf.sigmoid(y)
        elif(mapping_mode=='IM'):
            return tf.log(tf.sigmoid(y))
            # return y
        else:
            raise ValueError("Unexpected mode for target {}".format(mapping_mode))