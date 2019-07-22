# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/12/10
@note : Script to extract log specture from a list
'''

import os
import sys
import numpy as np
from scipy import signal
import soundfile
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'../'))
from tf_regression.utils.cal_lps import wavConfig as config
from tf_regression.utils.cal_lps import calc_lps_wapper2,calc_sp

def make_sequence_example(name,inputs,inputs_angle,outputs=None,noises=None):
    context=tf.train.Features(feature={'name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[name]))})
    inputs_features_amp = tf.train.FeatureList(feature=[
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs])
    inputs_features_ang = tf.train.FeatureList(feature=[
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs_angle])
    if(outputs is not None):
        output_features_amp = tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=output_))
            for output_ in outputs])
        output_features_noises = tf.train.FeatureList(feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=output_))
            for output_ in noises])
        gather_features = tf.train.FeatureLists(feature_list={
            'inputs':inputs_features_amp,
            'inputs_ang':inputs_features_ang,
            'outputs':output_features_amp,
            # 'outputs_ang':output_features_ang,
            'noises':output_features_noises,
        })
    else:
        gather_features = tf.train.FeatureLists(feature_list={
            'inputs':inputs_features_amp,
            'inputs_ang':inputs_features_ang,
        })
    return tf.train.SequenceExample(context=context, feature_lists=gather_features)

if __name__ == '__main__':
    if(len(sys.argv)!=4):
        raise Exception("Usage : python3 {} <file-list> <tf-record> <scale-dir>".format(sys.argv[0]))
    else:
        print("{}".format(" ".join(sys.argv)))
    tfrecord=sys.argv[2]
    scale_dir=sys.argv[3]

    fileList=[]
    with open(sys.argv[1],'r') as f:
        for line in f:
            fileList.append(line.strip().split())
    
    cfg=config();debugFlag=True
    speech_mean=np.zeros(int(cfg.n_window/2+1),dtype=np.float128)
    speech_var=np.zeros(int(cfg.n_window/2+1),dtype=np.float128)
    mix_mean=np.zeros(int(cfg.n_window/2+1),dtype=np.float128)
    mix_var=np.zeros(int(cfg.n_window/2+1),dtype=np.float128)
    counter=0
    with tf.python_io.TFRecordWriter(tfrecord) as writer:
        for inx,f in enumerate(fileList):
            print("Extract feature for {}".format(f))

            speech_fname=os.path.basename(f[1]).split(".")[0]
            (speech_audio, fs) = soundfile.read(f[1])
            assert fs==cfg.sample_rate, "Unmatched samplerate {}".format(fs)
            speech_x, speech_x_angle = calc_lps_wapper2(speech_audio, cfg=cfg)

            mix_fname=os.path.basename(f[2]).split(".")[0]
            (mix_audio, fs) = soundfile.read(f[2])
            assert fs==cfg.sample_rate, "Unmatched samplerate {}".format(fs)
            mix_x, mix_x_angle = calc_lps_wapper2(mix_audio, cfg=cfg)
            
            noise_audio = mix_audio - speech_audio
            noise_x, _ =calc_lps_wapper2(noise_audio, cfg=cfg)

            if(inx==0 and debugFlag):
                np.save(os.path.join(scale_dir,speech_fname+'.npy'),speech_x)
                np.save(os.path.join(scale_dir,mix_fname+'.npy'),mix_x)

            ex=make_sequence_example(mix_fname.encode(),mix_x,mix_x_angle,speech_x,noise_x)
            writer.write(ex.SerializeToString())

            speech_mean+=speech_x.sum(0)
            speech_var+=np.power(speech_x,2.0).sum(0)
            mix_mean+=mix_x.sum(0)
            mix_var+=np.power(mix_x,2.0).sum(0)
            counter+=speech_x.shape[0]
    
    print("Finish extracting features")
    speech_mean=speech_mean/counter;speech_var=np.sqrt((speech_var/counter-speech_mean**2))
    mix_mean=mix_mean/counter;mix_var=np.sqrt((mix_var/counter-mix_mean**2))

    np.save(os.path.join(scale_dir,'speech.mean.npy'),speech_mean)
    np.save(os.path.join(scale_dir,'speech.std.npy'),speech_var)
    np.save(os.path.join(scale_dir,'mixed.mean.npy'),mix_mean)
    np.save(os.path.join(scale_dir,'mixed.std.npy'),mix_var)
