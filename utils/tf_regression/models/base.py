# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

class Model(object):

    def __init__(self,sess, name='BaseModel'):
        self.name = name
        self.sess = sess
        self.MOVING_AVERAGE_DECAY = 0.999
        self.NUM_UPDATES = 500

    def save(self, save_dir, name, step):
        model_name = name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(max_to_keep=10)
        self.saver.save(self.sess,
                os.path.join(save_dir, model_name),
                global_step=step)

    def load(self, save_dir, model_file=None, moving_average=False):
        if not os.path.exists(save_dir):
            tf.logging.info('[!] Checkpoints path does not exist...')
            return False
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        
        tf.logging.info('[*] Reading checkpoints...')
        if moving_average:
            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                self.MOVING_AVERAGE_DECAY,self.NUM_UPDATES)
            variables_to_restore = variable_averages.variables_to_restore()
            for v in list(variables_to_restore.keys()):
                if v.find(self.name)<0:
                    del variables_to_restore[v]
            saver = tf.train.Saver(variables_to_restore)
        else:
            saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(save_dir, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True
