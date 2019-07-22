# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/12/11
@note : Main script to train a tensorflow model for regression task
        The code is writtern following GAN project under 
        /nobackup/f1/asr/wulong/rsrgan/DNN-FM-DNN-AC-fintune-40-20-40/io_funcs
'''

from __future__ import absolute_import
from __future__ import print_function
import argparse
import datetime
import time
import os
import sys
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import numpy as np
import pprint
import tensorflow as tf
import soundfile

sys.path.append(os.path.dirname(sys.path[0]))
from models.regress_trainer import RegressTrainer
from models.progressive_learning_trainer import ProgressiveLearningTrainer
from utils.misc import show_all_variables
from utils.transforms import preprocess_wapper_pl,inv_znorm
from utils.cal_lps import calc_lps_wapper2,recover_lps2wav_wapper,wavConfig
from io_funcs.tfrecords_dataset_pl import get_batch,get_num_samples

def logging(cur_it,total_it,message,loss_names,loss_nums,t):
    s="{}/{}, ({}), ".format(cur_it,total_it,message)
    for n in range(len(loss_names)):
        s+="{}={:.2e}, ".format(loss_names[n],loss_nums[n])
    s+="{}={:.2f}".format("time",t)
    print(s)

def run_one_iter(sess,batch_per_iter,model,names,\
                inputs,labels,pl_labels,lengths,it,\
                input_fn=None,output_fn=None,updateOp=True):
    losses_num=list()
    for batch_iter in range(batch_per_iter):
        # queue_names,queue_inputs,queue_labels,queue_lengths=batch_queue.get()
        queue_names,queue_inputs,queue_labels,queue_labels_pl,queue_lengths=sess.run([names,inputs,labels,pl_labels,lengths])
        # For debug
        # debug_item_index=np.argwhere(queue_names == 'P09_S03_U06_00003_10'.encode())
        # if(debug_item_index.size>0):
        #     print(queue_names[debug_item_index])
        #     print(queue_inputs[debug_item_index])
        #     print(queue_labels[debug_item_index])
        #     print(queue_lengths[debug_item_index])
        #     exit("Finish debug")
        if(queue_inputs.shape[0]) != FLAGS.batch_size:
            tf.logging.error("Unexpected input batch size {}".format(queue_inputs.shape))
        if(input_fn is not None):queue_inputs=input_fn(queue_inputs)
        if(output_fn is not None):queue_labels=output_fn(queue_labels)
        if(updateOp):
            opts,loss,loss_ratio,lr=sess.run([model.opts,model.loss,model.loss_ratio,model.lr],
                feed_dict={
                    model.inputs: queue_inputs,
                    model.labels:queue_labels,
                    model.pl_labels:queue_labels_pl,
                    model.lengths:queue_lengths,})
            losses_num.append((loss,loss_ratio,lr))
        else:
            loss,loss_ratio,=sess.run([model.loss,model.loss_ratio,],
                feed_dict={
                    model.inputs: queue_inputs,
                    model.labels:queue_labels,
                    model.pl_labels:queue_labels_pl,
                    model.lengths:queue_lengths,})
            losses_num.append((loss,loss_ratio))
            # print(queue_labels);exit(1)
            # print(debughook1);exit(1)
    return np.array(losses_num).mean(0)

def train():
    print("================Begin Training================")
    if(FLAGS.train_num_samples>0):
        tr_num_samples=FLAGS.train_num_samples
    else:
        tr_num_samples = get_num_samples(FLAGS.train_data,FLAGS.batch_size,\
                    FLAGS.input_dim,FLAGS.output_dim,infer=False)
        FLAGS.train_num_samples=tr_num_samples
    if(FLAGS.valid_num_samples>0):
        cv_num_samples=FLAGS.valid_num_samples
    else:
        cv_num_samples = get_num_samples(FLAGS.valid_data,FLAGS.batch_size,\
                    FLAGS.input_dim,FLAGS.output_dim,infer=False)
        FLAGS.valid_num_samples=cv_num_samples
    train_batch_per_iter = 1;valid_batch_per_iter = int(cv_num_samples/FLAGS.batch_size)
    num_iters = int(tr_num_samples*FLAGS.epochs/FLAGS.batch_size/train_batch_per_iter)
    check_interval = int(tr_num_samples/FLAGS.batch_size/train_batch_per_iter/FLAGS.check_times_one_epoch)
    print("================Training Info================")
    tf.logging.info("Train sample num {}".format(tr_num_samples))
    tf.logging.info("Train batch size {}".format(FLAGS.batch_size))
    tf.logging.info("Train batch per iter {}".format(train_batch_per_iter))
    tf.logging.info("Validation sample num {}".format(cv_num_samples))
    tf.logging.info("Validation batch size {}".format(FLAGS.batch_size))
    tf.logging.info("Validation batch per iter {}".format(valid_batch_per_iter))
    tf.logging.info("Epochs {}".format(FLAGS.epochs))
    tf.logging.info("Num of iters {}".format(num_iters))
    tf.logging.info("Check inerval is {}".format(check_interval))
    print("=============================================")

    input_mean=np.load(FLAGS.input_mean)
    input_std=np.load(FLAGS.input_std)
    output_mean=np.load(FLAGS.output_mean)
    output_std=np.load(FLAGS.output_std)
    def pre_fn(x,y,noise,y_pl,noise_pl):return preprocess_wapper_pl(x,y,noise,y_pl,noise_pl,input_mean,input_std,FLAGS.mapping_mode)

    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.name_scope('inputs'):
                tr_name,tr_inputs,tr_labels,tr_labels_pl,tr_lengths=get_batch(
                    filename=FLAGS.train_data,
                    batch_size=FLAGS.batch_size,
                    input_dim=FLAGS.input_dim,
                    output_dim=FLAGS.output_dim,
                    left_context=FLAGS.left_context,
                    right_context=FLAGS.right_context,
                    num_epochs=FLAGS.epochs+1, # add one epoch for avoiding accident 
                    infer=False,
                    num_threads=FLAGS.num_threads,
                    transform_fn=pre_fn,)
                cv_name,cv_inputs,cv_labels,cv_labels_pl,cv_lengths=get_batch(
                    filename=FLAGS.valid_data,
                    batch_size=FLAGS.batch_size,
                    input_dim=FLAGS.input_dim,
                    output_dim=FLAGS.output_dim,
                    left_context=FLAGS.left_context,
                    right_context=FLAGS.right_context,
                    num_epochs=-1,
                    infer=False,
                    num_threads=FLAGS.num_threads,
                    transform_fn=pre_fn,)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            with tf.name_scope('regression_model'):
                print("=======================================================")
                print("|                Build Train model                    |")
                print("=======================================================")
                tr_model = ProgressiveLearningTrainer(sess, FLAGS, cross_validation=False)
                # tr_model and val_model should share variables
                print("=======================================================")
                print("|           Build Cross-Validation model              |")
                print("=======================================================")
                tf.get_variable_scope().reuse_variables()
                cv_model = ProgressiveLearningTrainer(sess, FLAGS, cross_validation=True)
            show_all_variables()
            # initialize
            init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
            tf.logging.info("Initializing variables ...")
            sess.run(init)

            # If you stop training and now want to train again, just load the origin model
            if tr_model.load(FLAGS.model_dir, moving_average=False):
                tf.logging.info("[*] Load SUCCESS")
            else:
                tf.logging.info("[!] Begin a new model.")
            sys.stdout.flush()

            # train_batch_queue = queue.Queue(train_batch_per_iter)
            # valid_batch_queue = queue.Queue(valid_batch_per_iter)
            loss_names_cv=["mse","mse_ratio"];loss_names_tr=["mse","mse_ratio","lr"]
            start_time=time.time()
            cv_loss=run_one_iter(sess,valid_batch_per_iter,\
                cv_model,cv_name,cv_inputs,cv_labels,cv_labels_pl,cv_lengths,\
                0,None,None,False)
            end_time=time.time()
            logging(0,num_iters,"SUMMARY CV LOSS",loss_names_cv,cv_loss,end_time-start_time)
            tr_losses=[];best_loss=1e12
            for it in range(num_iters):
                try:
                    start_time=time.time()
                    tr_loss=run_one_iter(sess,train_batch_per_iter,\
                        tr_model,tr_name,tr_inputs,tr_labels,tr_labels_pl,tr_lengths,\
                        it+1,None,None,True)
                    tr_losses.append(tr_loss)
                    end_time=time.time()
                    logging(it+1,num_iters,"TRAIN LOSS",loss_names_tr,tr_loss,end_time-start_time)
                    if((it+1)%check_interval==0):
                        tr_losses=np.array(tr_losses).mean(0)
                        logging(it+1,num_iters,"SUMMARY TRAIN LOSS",loss_names_cv,tr_losses,0.0)

                        start_time=time.time()
                        cv_loss=run_one_iter(sess,valid_batch_per_iter,\
                            cv_model,cv_name,cv_inputs,cv_labels,cv_labels_pl,cv_lengths,\
                            it+1,None,None,False)
                        end_time=time.time()
                        logging(it+1,num_iters,"SUMMARY CV LOSS",loss_names_cv,cv_loss,end_time-start_time)

                        if(best_loss>cv_loss[0]):
                            best_loss=cv_loss[0]
                            tr_model.save(FLAGS.model_dir,"iter_{}_{:.2e}".format(it+1,cv_loss[0]),it+1)
                        tr_losses=[]
                except tf.errors.OutOfRangeError:
                    tf.logging.error("Out of range occured when training.")
                    break
            sess.close()
    print("================Finish Training================")

def infer():
    print("================Begin Infer================")
    infer_num_samples = get_num_samples(FLAGS.infer_data,1,\
            FLAGS.input_dim,FLAGS.output_dim,infer=True)
    if(not os.path.exists(FLAGS.infer_out_dir)):
        os.makedirs(FLAGS.infer_out_dir)
        tf.logging.info('Sucessfully make dir {}'.format(FLAGS.infer_out_dir))
    else:
        tf.logging.info("Dir {} already exists".format(FLAGS.infer_out_dir))

    wavCfg=wavConfig()
    input_mean=np.load(FLAGS.input_mean)
    input_std=np.load(FLAGS.input_std)
    output_mean=np.load(FLAGS.output_mean)
    output_std=np.load(FLAGS.output_std)
    def pre_fn(x,y,noise,y_pl,noise_pl):return preprocess_wapper_pl(x,None,None,None,None,input_mean,input_std,FLAGS.mapping_mode)

    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.name_scope('inputs'):
                infer_names,infer_inputs,infer_angles,infer_lengths=get_batch(
                    filename=FLAGS.infer_data,
                    batch_size=1,
                    input_dim=FLAGS.input_dim,
                    output_dim=FLAGS.output_dim,
                    left_context=FLAGS.left_context,
                    right_context=FLAGS.right_context,
                    num_epochs=1,
                    infer=True,
                    num_threads=FLAGS.num_threads,
                    transform_fn=pre_fn,)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            with tf.name_scope('regression_model'):
                print("=======================================================")
                print("|                Build Infer model                    |")
                print("=======================================================")
                infer_model = ProgressiveLearningTrainer(sess, FLAGS, cross_validation=True,infer=True)
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            tf.logging.info("Initializing variables ...")
            sess.run(init)
            # The moving_average can be true only if the num of epochs is big enough
            # maybe at least larger then 5 
            if infer_model.load(FLAGS.model_dir, moving_average=True):
                tf.logging.info("[*] Load SUCCESS")
            else:
                raise FileNotFoundError("Load model in {} failed".format*FLAGS.model_dir)
            sys.stdout.flush()

            for sample_ind in range(infer_num_samples):
                queue_names,queue_inputs,queue_angles,queue_lengths=\
                    sess.run([infer_names,infer_inputs,infer_angles,infer_lengths])
                pred,lengths=sess.run([infer_model.net_output,infer_model.lengths],
                    feed_dict={infer_model.inputs: queue_inputs,
                    infer_model.lengths:queue_lengths,})
                
                infer_base_name=queue_names[0].decode("utf-8")+'.wav'
                infer_save_path=os.path.join(FLAGS.infer_out_dir,infer_base_name)
                tf.logging.info("Writing file {} to {}".format(infer_base_name,infer_save_path))
                if FLAGS.mapping_mode=='DM':
                    pred = pred[0,0:lengths[0],:]
                    # pred = inv_znorm(pred[0,0:lengths[0],:],output_mean,output_std)
                elif FLAGS.mapping_mode=='IRM':
                    # pred_lps=log(exp(inputs)*pred)
                    org_inputs=inv_znorm(queue_inputs[:,:,FLAGS.input_dim*FLAGS.left_context:FLAGS.input_dim*(FLAGS.left_context+1)],input_mean,input_std)
                    pred = org_inputs+np.log(pred)
                    pred = pred[0,0:lengths[0],:]
                    # pred = inv_znorm(pred,output_mean,output_std)
                elif FLAGS.mapping_mode=='IM':
                    # pred_lps=log(exp(inputs)*exp(pred))
                    org_inputs=inv_znorm(queue_inputs[:,:,FLAGS.input_dim*FLAGS.left_context:FLAGS.input_dim*(FLAGS.left_context+1)],input_mean,input_std)
                    pred = org_inputs+pred
                    pred = pred[0,0:lengths[0],:]
                    # pred = inv_znorm(pred,output_mean,output_std)
                else:
                    raise ValueError("Unexpected mapping mode {}".format(FLAGS.mapping_mode))
                audio=recover_lps2wav_wapper(pred,queue_angles[0],wavCfg,norm=True)
                soundfile.write(file=infer_save_path,data=audio,samplerate=wavCfg.sample_rate)

def eager_infer():
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

    print("================Begin Eager Infer================")
    wavCfg=wavConfig()
    if((FLAGS.num_samples_per_seg-wavCfg.n_window)%(wavCfg.n_window-wavCfg.n_overlap)!=0):
        raise ValueError("It is highly recommanded that the num_samples_per_seg should be framed without the remainder")
    else:
        num_frames_per_seg=(FLAGS.num_samples_per_seg-wavCfg.n_window)//(wavCfg.n_window-wavCfg.n_overlap)
    
    input_mean=np.load(FLAGS.input_mean)
    input_std=np.load(FLAGS.input_std)
    output_mean=np.load(FLAGS.output_mean)
    output_std=np.load(FLAGS.output_std)
    
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.name_scope('inputs'):
                infer_audio,fs=soundfile.read(FLAGS.input_wav)
                infer_inputs,infer_angles=calc_lps_wapper2(infer_audio,cfg=wavCfg)
                infer_inputs=(infer_inputs-input_mean)/input_std

                infer_inputs=(infer_inputs.reshape(1,*(infer_inputs.shape))).astype(np.float32)
                infer_angles=(infer_angles.reshape(1,*(infer_angles.shape))).astype(np.float32)
                infer_inputs=tf.constant(infer_inputs)
                queue_inputs=splice_feats(infer_inputs,FLAGS.left_context,FLAGS.right_context)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True  

        with tf.Session(config=config) as sess:
            with tf.name_scope('regression_model'):
                print("=======================================================")
                print("|                Build Infer model                    |")
                print("=======================================================")
                infer_model = ProgressiveLearningTrainer(sess, FLAGS, cross_validation=True,infer=True)
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            tf.logging.info("Initializing variables ...")
            sess.run(init)
            # The moving_average can be true only if the num of epochs is big enough
            # maybe at least larger then 5 
            if infer_model.load(FLAGS.model_dir, moving_average=True):
                tf.logging.info("[*] Load SUCCESS")
            else:
                raise FileNotFoundError("Load model in {} failed".format(FLAGS.model_dir))
            sys.stdout.flush()  

            queue_inputs=sess.run(queue_inputs)
            for start_index in range(0,queue_inputs.shape[1],num_frames_per_seg):
                end_index=start_index+num_frames_per_seg if start_index+num_frames_per_seg<queue_inputs.shape[1] else queue_inputs.shape[1]
                infer_lengths=np.array([end_index-start_index,])
                pred_temp=sess.run(infer_model.net_output,
                    feed_dict={infer_model.inputs: queue_inputs[:,start_index:end_index,:],
                    infer_model.lengths:infer_lengths,})
                if(start_index==0):pred=pred_temp
                else:pred=np.concatenate((pred,pred_temp),axis=1)
            
            infer_save_path=FLAGS.output_wav
            tf.logging.info("Writing file to {}".format(infer_save_path))
            if FLAGS.mapping_mode=='DM':
                pred = pred[0,0:queue_inputs.shape[1],:]
                # pred = inv_znorm(pred[0,0:lengths[0],:],output_mean,output_std)
            elif FLAGS.mapping_mode=='IRM':
                # pred_lps=log(exp(inputs)*pred)
                org_inputs=inv_znorm(queue_inputs[:,:,FLAGS.input_dim*FLAGS.left_context:FLAGS.input_dim*(FLAGS.left_context+1)],input_mean,input_std)
                pred = org_inputs+np.log(pred)
                pred = pred[0,0:queue_inputs.shape[1],:]
                # pred = inv_znorm(pred,output_mean,output_std)
            elif FLAGS.mapping_mode=='IM':
                # pred_lps=log(exp(inputs)*exp(pred))
                org_inputs=inv_znorm(queue_inputs[:,:,FLAGS.input_dim*FLAGS.left_context:FLAGS.input_dim*(FLAGS.left_context+1)],input_mean,input_std)
                pred = org_inputs+pred
                pred = pred[0,0:queue_inputs.shape[1],:]
                # pred = inv_znorm(pred,output_mean,output_std)
            else:
                raise ValueError("Unexpected mapping mode {}".format(FLAGS.mapping_mode))
            # Note: If you want to infer a single wav, using norm may be better
            # If you want to infer multi-channel wavs and may take beamform in the future, norm leads to performance degradation
            audio=recover_lps2wav_wapper(pred,infer_angles[0],wavCfg,norm=False)
            audio=audio[0:infer_audio.shape[0]]
            soundfile.write(file=infer_save_path,data=audio,samplerate=wavCfg.sample_rate)

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim",
        type=int,required=True,
        help='input dim')
    parser.add_argument(
        "--output_dim",
        type=int,required=True,
        help='output dim')
    parser.add_argument(
        "--left_context",
        type=int,required=True,
        help='input expansion through left context')
    parser.add_argument(
        "--right_context",
        type=int,required=True,
        help='input expansion through right context')
    parser.add_argument(
        "--num_hidden_layers",
        type=int,required=True,
        help='num of hidden blstm layers')
    parser.add_argument(
        "--cell_dim",
        type=int,required=True,
        help='cell dim')    
    parser.add_argument(
        "--proj_dim",
        type=int,required=True,
        help='projection dim')
    parser.add_argument(
        "--recur_dim",
        type=int,required=True,
        help='recurrent dim')
    parser.add_argument(
        "--mapping_mode",
        type=str,required=True,
        help='The feature mapping mode(DM/IRM/IM)')
    parser.add_argument(
        "--net_type",
        type=str,required=True,
        help='Net type\nCurrently support DNN/BLSTM'
    )
    parser.add_argument(
        "--model_dir",
        type=str,required=True,
        help='model dir')
    parser.add_argument(
        "--num_threads",
        type=int,required=False,
        default=4,help='num of threads to load data')
    parser.add_argument(
        "--input_mean",
        type=str,required=True,
        help='The mean of input')
    parser.add_argument(
        "--input_std",
        type=str,required=True,
        help='The std of input')
    parser.add_argument(
        "--output_mean",
        type=str,required=True,
        help='The mean of output')
    parser.add_argument(
        "--output_std",
        type=str,required=True,
        help='The std of output')

    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument(
        "--batch_size",
        type=int,required=True,
        help='batch_size')
    parser_train.add_argument(
        "--epochs",
        type=int,required=True,
        help='num of epochs')
    parser_train.add_argument(
        "--lr",
        type=float,required=True,
        help='learning rate')
    parser_train.add_argument(
        "--start_step",
        type=int,required=False,default=0,
        help='The start step index for training')
    parser_train.add_argument(
        "--train_data",
        type=str,required=True,
        help='train data tfrecords')
    parser_train.add_argument(
        "--valid_data",
        type=str,required=True,
        help='validation data tfrecords')
    parser_train.add_argument(
        "--train_num_samples",
        type=int,required=False,
        default=-1,
        help="Input the num of batches for one epoch in training set if you know.\
             Infering this number may cost a lot of time")
    parser_train.add_argument(
        "--valid_num_samples",
        type=int,required=False,
        default=-1,
        help="Input the num of batches for one epoch in validate set if you know.\
             Infering this number may cost a lot of time")
    parser_train.add_argument(
        "--check_times_one_epoch",
        type=int,required=False,
        default=1,
        help="The num of times of evaluating CV and storing net in one epoch")

    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument(
        "--infer_data",
        type=str,required=True,
        help='validation data tfrecords')
    parser_infer.add_argument(
        "--infer_out_dir",
        type=str,required=True,
        help='infer out dir for storing wav files')

    parser_eager = subparsers.add_parser('eager_infer')
    parser_eager.add_argument(
        "--input_wav",
        type=str,required=True,
        help='input wav file'
    )
    parser_eager.add_argument(
        "--output_wav",
        type=str,required=True,
        help='output wav file'
    )
    parser_eager.add_argument(
        "--num_samples_per_seg",
        type=int,required=False,
        default=1920000,
        help='process the wav file in segs\n\
            default is 120s*16kHz'
    )

    print("======={}=======".format(sys.argv[0]))
    global FLAGS;FLAGS = parser.parse_args()
    pprint.pprint(FLAGS.__dict__)
    print("=======Mode {}=======".format(FLAGS.mode))

    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'infer':
        FLAGS.batch_size=1
        infer()
    elif FLAGS.mode == 'eager_infer':
        FLAGS.batch_size=1
        eager_infer()
    else:
        raise Exception('Unexpected mode {}'.format(FLAGS.mode)) 
