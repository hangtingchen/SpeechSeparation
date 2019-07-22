#!/bin/bash

# author : chenhangting
# date : 2018/12/11

# The script is to train an example lstm using tensorflow.

. ./path_tf.sh
set -ue

approach=DM
net=DNN

## step 0
if [ -d exp/tf_test_${net}_$approach ];then
    rm -r exp/tf_test_${net}_$approach
fi

## step 1
TF_CPP_MIN_LOG_LEVEL=2 \
    python3 -u utils_cht2/tf_regression/run_regression.py \
        --input_dim 257 \
        --output_dim 257 \
        --left_context 3 \
        --right_context 3 \
        --num_hidden_layers 3 \
        --cell_dim 2048 \
        --proj_dim -1 \
        --recur_dim -1 \
        --mapping_mode $approach \
        --net_type $net \
        --model_dir exp/tf_test_${net}_$approach \
        --input_mean data/CHiME5_mixed_audio/S03/P09/feat/train/mixed.mean.npy \
        --input_std data/CHiME5_mixed_audio/S03/P09/feat/train/mixed.std.npy \
        --output_mean data/CHiME5_mixed_audio/S03/P09/feat/train/speech.mean.npy \
        --output_std data/CHiME5_mixed_audio/S03/P09/feat/train/speech.std.npy \
        train \
        --batch_size 64 \
        --epochs 10 \
        --lr 0.003 \
        --train_data data/CHiME5_mixed_audio/S03/P09/feat/train/train.tfrecords \
        --valid_data data/CHiME5_mixed_audio/S03/P09/feat/test/test.tfrecords \
        --check_times_one_epoch 2 \
        --train_num_samples 15949 \
        --valid_num_samples 495

## step 2
TF_CPP_MIN_LOG_LEVEL=2 \
    python3 -u utils_cht2/tf_regression/run_regression.py \
        --input_dim 257 \
        --output_dim 257 \
        --left_context 3 \
        --right_context 3 \
        --num_hidden_layers 3 \
        --cell_dim 2048 \
        --proj_dim -1 \
        --recur_dim -1 \
        --mapping_mode $approach \
        --net_type $net \
        --model_dir exp/tf_test_${net}_$approach \
        --input_mean data/CHiME5_mixed_audio/S03/P09/feat/train/mixed.mean.npy \
        --input_std data/CHiME5_mixed_audio/S03/P09/feat/train/mixed.std.npy \
        --output_mean data/CHiME5_mixed_audio/S03/P09/feat/train/speech.mean.npy \
        --output_std data/CHiME5_mixed_audio/S03/P09/feat/train/speech.std.npy \
        infer \
        --infer_data data/CHiME5_mixed_audio/S03/P09/feat/test/test.tfrecords \
        --infer_out_dir exp/tf_test_${net}_${approach}/infer

## step 3
pesq=~/usr/pesq/P862/Software/pesq
if [ -f exp/tf_test_${net}_${approach}/pesq_report.txt ];then
    rm exp/tf_test_${net}_${approach}/pesq_report.txt
fi
touch exp/tf_test_${net}_${approach}/pesq_report.txt
cat exp/ss1_training/S03/P09/audio_list.test.txt | while read line;do
    org_audio=`echo $line | awk '{print $2}'`
    mixed_audio=`echo $line | awk '{print $3}'`
    base_mixed_audio=`basename $mixed_audio`
    infer_audio=exp/tf_test_${net}_${approach}/infer/${base_mixed_audio}
    mixed_audio_pesq=`$pesq +16000 ${org_audio} ${mixed_audio} | tail -1 | awk -F= '{print $2}'`
    infer_audio_pesq=`$pesq +16000 ${org_audio} ${infer_audio} | tail -1 | awk -F= '{print $2}'`
    echo "${base_mixed_audio}${mixed_audio_pesq}${infer_audio_pesq}"
    echo "${base_mixed_audio}${mixed_audio_pesq}${infer_audio_pesq}" >> exp/tf_test_${net}_${approach}/pesq_report.txt
done
python3 -u utils_cht2/tf_regression/pesq2report.py exp/tf_test_${net}_${approach}/pesq_report.txt exp/tf_test_${net}_${approach}/pesq_report.report

## repeat step 2 and 3 for pure test
## step 2
TF_CPP_MIN_LOG_LEVEL=2 \
    python3 -u utils_cht2/tf_regression/run_regression.py \
        --input_dim 257 \
        --output_dim 257 \
        --left_context 3 \
        --right_context 3 \
        --num_hidden_layers 3 \
        --cell_dim 2048 \
        --proj_dim -1 \
        --recur_dim -1 \
        --mapping_mode $approach \
        --net_type $net \
        --model_dir exp/tf_test_${net}_$approach \
        --input_mean data/CHiME5_mixed_audio/S03/P09/feat/train/mixed.mean.npy \
        --input_std data/CHiME5_mixed_audio/S03/P09/feat/train/mixed.std.npy \
        --output_mean data/CHiME5_mixed_audio/S03/P09/feat/train/speech.mean.npy \
        --output_std data/CHiME5_mixed_audio/S03/P09/feat/train/speech.std.npy \
        infer \
        --infer_data data/CHiME5_mixed_audio/S03/P09/feat/pure_test/pure_test.tfrecords \
        --infer_out_dir exp/tf_test_${net}_${approach}/infer_pure

## step 3
pesq=~/usr/pesq/P862/Software/pesq
if [ -f exp/tf_test_${net}_${approach}/pesq_report_pure.txt ];then
    rm exp/tf_test_${net}_${approach}/pesq_report_pure.txt
fi
touch exp/tf_test_${net}_${approach}/pesq_report_pure.txt
cat exp/ss1_training/S03/P09/audio_list.pure_test.txt | while read line;do
    org_audio=`echo $line | awk '{print $2}'`
    mixed_audio=`echo $line | awk '{print $3}'`
    base_mixed_audio=`basename $mixed_audio`
    infer_audio=exp/tf_test_${net}_${approach}/infer_pure/${base_mixed_audio}
    mixed_audio_pesq=`$pesq +16000 ${org_audio} ${mixed_audio} | tail -1 | awk -F= '{print $2}'`
    infer_audio_pesq=`$pesq +16000 ${org_audio} ${infer_audio} | tail -1 | awk -F= '{print $2}'`
    echo "${base_mixed_audio}${mixed_audio_pesq}${infer_audio_pesq}"
    echo "${base_mixed_audio}${mixed_audio_pesq}${infer_audio_pesq}" >> exp/tf_test_${net}_${approach}/pesq_report_pure.txt
done
python3 -u utils_cht2/tf_regression/pesq2report.py exp/tf_test_${net}_${approach}/pesq_report_pure.txt exp/tf_test_${net}_${approach}/pesq_report_pure.report

echo "Everything is ok"
