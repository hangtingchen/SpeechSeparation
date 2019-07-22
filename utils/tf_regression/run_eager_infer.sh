#!/bin/bash

# author : chenhangting
# date : 2018/12/19

# The script is to run eager infer mode for a model

. ./path_tf.sh
set -ue

approach=DM
net=DNN

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
        eager_infer \
        --input_wav exp/mix/P09_S03_U06_00003_orig.wav \
        --output_wav exp/mix/P09_S03_U06_00003_orig_${net}_${approach}.wav