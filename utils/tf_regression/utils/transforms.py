# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/12/12
@note : Contains various transforms
'''

import tensorflow as tf
import numpy as np

def znorm(x,mean,std):
    assert x.shape[-1]==mean.shape[-1] \
        and x.shape[-1]==std.shape[-1], \
        "Size not match for x {}, mean {}, std {}".format(x.shape,mean.shape,std.shape)
    return (x-mean)/std

def inv_znorm(x,mean,std):
    assert x.shape[-1]==mean.shape[-1] \
        and x.shape[-1]==std.shape[-1], \
        "Size not match for x {}, mean {}, std {}".format(x.shape,mean.shape,std.shape)
    return x*std+mean

def znorm_tf(x,mean,std):
    mean_tf=tf.constant(mean.astype(np.float32))
    std_tf=tf.constant(std.astype(np.float32))
    return (x-mean_tf)/std_tf

def inv_znorm_tf(x,mean,std):
    mean_tf=tf.constant(mean.astype(np.float32))
    std_tf=tf.constant(std.astype(np.float32))
    return x*std_tf+mean_tf

def mapping_target_fn(x,y,noise,mode):
    '''
    How to map features.\n
    Input x,y should be log amplitude spectral.(Please note here we use amplitude)\n
    Mode = DM,IRM,IM\n
    Here we discribe these three modes\n
    F() means the network mapping\n
    DM is direct mapping : LOSS = (F(x)-y)^2
    IRM is using ideal ratio mask : LOSS = (F(x)-exp(2y)/exp(2x))^2
    IM is indirect mapping : LOSS = (log(F(x))-(2y-2x))^2
    '''
    if(mode=='DM'):
        return y
    elif(mode=='IRM'):
        # we now directly calculate the irm from noise+specch instead of mixed audio
        # return tf.clip_by_value(tf.exp(y-x),0.0,1.0)
        return tf.exp(y)/(tf.exp(y)+tf.exp(noise))
    elif(mode=='IM'):
        return tf.log(tf.exp(y)/(tf.exp(y)+tf.exp(noise)))
    else:
        raise ValueError("Unexpected mode for target {}".format(mode))

def preprocess_wapper(x,y,noise,x_mean,x_std,mode):
    if(y is not None):
        tar=mapping_target_fn(x,y,noise,mode)
    else:
        tar=None
    x=znorm_tf(x,x_mean,x_std)
    return x,tar

def preprocess_wapper_pl(x,y,noise,y_pl,noise_pl,x_mean,x_std,mode):
    if(y is not None):
        tar1=mapping_target_fn(x,y,noise,mode)
        tar2=mapping_target_fn(x,y_pl,noise_pl,mode)
    else:
        tar1=None
        tar2=None
    x=znorm_tf(x,x_mean,x_std)
    return x,tar1,tar2
